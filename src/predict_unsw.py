"""Σύστημα Α – πρόβλεψη για νέες ροές δικτύου (UNSW-NB15, 8 χαρακτηριστικά).

Χρήση από Python:
    from predict_unsw import UnswDetector
    detector = UnswDetector()                 # φορτώνει δίκτυο, preprocessor και ρυθμίσεις από τον φάκελο models/
    results = detector.predict(df)            # df: DataFrame με (τουλάχιστον) τις 8 στήλες εισόδου

Χρήση από τη γραμμή εντολών:
    python src/predict_unsw.py data/new_flows.csv -o data/new_flows_predictions.csv

Για κάθε ροή επιστρέφει:
    p_attack           πιθανότητα επίθεσης (0–1)
    is_attack          1 αν p_attack >= κατώφλι, αλλιώς 0
    attack_type        τύπος επίθεσης ("Normal" αν δεν είναι επίθεση)
    severity           None / Low / Medium / High
    expected_severity  αναμενόμενο βάρος σοβαρότητας (0.33–1.0), δεδομένου ότι είναι επίθεση
    risk_score         100 · p_attack · expected_severity  (0–100)

Κανόνες σοβαρότητας (παράμετρος severity_rule):
    "expected_tuned"    προεπιλογή (κανόνας Γ, notebook 12): από την αναμενόμενη σοβαρότητα,
                        με όρια ρυθμισμένα στο validation. Η σοβαρότητα ΔΕΝ προκύπτει από τον τύπο.
    "expected_natural"  κανόνας Β: ίδια ποσότητα, όρια στα μέσα των βαρών (0.495 και 0.83).
    "from_type"         κανόνας Α: η σοβαρότητα που αντιστοιχεί στον προβλεπόμενο τύπο.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SEVERITY_RULES = ("expected_tuned", "expected_natural", "from_type")


class UnswDetector:
    """Φορτώνει μία φορά το δίκτυο και τις ρυθμίσεις και κάνει προβλέψεις για νέες ροές."""

    def __init__(self, models_dir=DEFAULT_MODELS_DIR):
        import keras  # εισάγεται εδώ ώστε το αρχείο να διαβάζεται και χωρίς TensorFlow

        models_dir = Path(models_dir)
        with open(models_dir / "unsw_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(models_dir / "unsw_8num_tuned_config.json", encoding="utf-8") as f:
            config = json.load(f)
        with open(models_dir / "unsw_decision_rule.json", encoding="utf-8") as f:
            rule = json.load(f)

        self.features = list(config["features"])
        self.attack_classes = list(meta["attack_classes"])
        self.severity_classes = list(meta["severity_classes"])
        self.severity_weights = np.array([meta["severity_weight"][c] for c in self.severity_classes], dtype="float64")
        self.severity_of_attack = np.array(
            [self.severity_classes.index(meta["severity_map"][c]) for c in self.attack_classes])
        self.threshold = float(rule["threshold"])
        self.cuts = {"expected_tuned": tuple(rule["severity_cuts_tuned"]),
                     "expected_natural": tuple(rule["severity_cuts_natural"])}

        self.preprocessor = joblib.load(models_dir / "unsw_preprocessor_8num_tuned.joblib")
        self.model = keras.models.load_model(models_dir / "unsw_multitask_8num_tuned.keras", compile=False)

    def _check_input(self, flows):
        if isinstance(flows, dict):
            flows = pd.DataFrame([flows])
        if not isinstance(flows, pd.DataFrame):
            raise TypeError("Η είσοδος πρέπει να είναι pandas DataFrame (ή dict για μία ροή).")
        missing = [c for c in self.features if c not in flows.columns]
        if missing:
            raise ValueError(f"Λείπουν στήλες εισόδου: {missing}. Απαιτούνται: {self.features}")
        X = flows[self.features].apply(pd.to_numeric, errors="coerce")
        bad = X.columns[X.isna().any()].tolist()
        if bad:
            raise ValueError(f"Κενές ή μη αριθμητικές τιμές στις στήλες: {bad}")
        negative = X.columns[(X < 0).any()].tolist()
        if negative:
            raise ValueError(f"Αρνητικές τιμές στις στήλες: {negative} (όλα τα χαρακτηριστικά είναι μη αρνητικά).")
        return X

    def predict(self, flows, severity_rule="expected_tuned", threshold=None):
        """Προβλέψεις για ένα DataFrame ροών. Επιστρέφει DataFrame με το ίδιο index."""
        if severity_rule not in SEVERITY_RULES:
            raise ValueError(f"Άγνωστος κανόνας σοβαρότητας: {severity_rule!r}. Επιλογές: {SEVERITY_RULES}")
        threshold = self.threshold if threshold is None else float(threshold)
        X = self._check_input(flows)
        if len(X) == 0:
            return pd.DataFrame(columns=["p_attack", "is_attack", "attack_type", "severity", "expected_severity", "risk_score"])

        Xt = self.preprocessor.transform(X).astype("float32")
        p_label, p_attack, p_sev = self.model.predict(Xt, batch_size=1024, verbose=0)
        p_label = np.asarray(p_label, dtype="float64").ravel()
        p_attack, p_sev = np.asarray(p_attack, dtype="float64"), np.asarray(p_sev, dtype="float64")

        is_attack = p_label >= threshold
        attack = np.where(is_attack, p_attack[:, 1:].argmax(1) + 1, 0)          # 0 = Normal

        p = p_sev[:, 1:]                                                         # Low, Medium, High
        expected = (p * self.severity_weights[1:]).sum(1) / np.clip(p.sum(1), 1e-9, None)
        if severity_rule == "from_type":
            severity = self.severity_of_attack[attack]
        else:
            c1, c2 = self.cuts[severity_rule]
            level = np.where(expected < c1, 1, np.where(expected < c2, 2, 3))
            severity = np.where(is_attack, level, 0)

        return pd.DataFrame({
            "p_attack": p_label,
            "is_attack": is_attack.astype(int),
            "attack_type": np.array(self.attack_classes)[attack],
            "severity": np.array(self.severity_classes)[severity],
            "expected_severity": expected,
            "risk_score": 100 * p_label * expected,
        }, index=X.index)


def main():
    parser = argparse.ArgumentParser(description="Σύστημα Α: προβλέψεις για ροές δικτύου από αρχείο CSV.")
    parser.add_argument("input_csv", help="CSV με τις 8 στήλες εισόδου (επιπλέον στήλες αγνοούνται)")
    parser.add_argument("-o", "--output", help="CSV εξόδου (προεπιλογή: <είσοδος>_predictions.csv)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--severity-rule", default="expected_tuned", choices=SEVERITY_RULES)
    parser.add_argument("--threshold", type=float, default=None, help="κατώφλι επίθεσης (προεπιλογή: από τις ρυθμίσεις)")
    args = parser.parse_args()

    flows = pd.read_csv(args.input_csv)
    detector = UnswDetector(args.models_dir)
    results = detector.predict(flows, severity_rule=args.severity_rule, threshold=args.threshold)
    output = args.output or str(Path(args.input_csv).with_name(Path(args.input_csv).stem + "_predictions.csv"))
    pd.concat([flows, results], axis=1).to_csv(output, index=False)

    print(f"Ροές: {len(results)} | επιθέσεις: {int(results['is_attack'].sum())} ({results['is_attack'].mean():.1%})")
    print(results["severity"].value_counts().reindex(detector.severity_classes, fill_value=0).to_string())
    print("Αποθηκεύτηκε:", output)


if __name__ == "__main__":
    main()
