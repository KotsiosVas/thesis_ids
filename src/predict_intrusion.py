"""Σύστημα Β – πρόβλεψη για νέες συνεδρίες (cybersecurity_intrusion_data, 9 είσοδοι).

Χρήση από Python:
    from predict_intrusion import IntrusionDetector
    detector = IntrusionDetector()            # φορτώνει δίκτυο, preprocessor και κανόνα από τον φάκελο models/
    results = detector.predict(df)            # df: DataFrame με (τουλάχιστον) τις 9 στήλες εισόδου

Χρήση από τη γραμμή εντολών:
    python src/predict_intrusion.py data/new_sessions.csv -o data/new_sessions_predictions.csv

Για κάθε συνεδρία επιστρέφει:
    p_attack      πιθανότητα επίθεσης από το νευρωνικό δίκτυο (0–1)
    is_attack     1 αν p_attack >= κατώφλι (προεπιλογή 0.5), αλλιώς 0
    rule_attack   1 αν ισχύει ο απλός κανόνας του δέντρου απόφασης (notebook B04), αλλιώς 0
    rule_reason   ποιος όρος του κανόνα ισχύει (κενό αν κανένας)

Σημειώσεις:
    - Το encryption_used = "None" σημαίνει «χωρίς κρυπτογράφηση». Αν η τιμή διαβαστεί ως κενή (NaN), μετατρέπεται σε "None".
    - Άγνωστες τιμές στις κατηγορικές στήλες (π.χ. browser που δεν υπήρχε στην εκπαίδευση) γίνονται δεκτές·
      το δίκτυο τις αντιμετωπίζει σαν καμία από τις γνωστές τιμές.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_THRESHOLD = 0.5          # επιλέχθηκε στο validation (notebook B03)


class IntrusionDetector:
    """Φορτώνει μία φορά το δίκτυο, τον preprocessor και τον κανόνα, και κάνει προβλέψεις για νέες συνεδρίες."""

    def __init__(self, models_dir=DEFAULT_MODELS_DIR):
        import keras  # εισάγεται εδώ ώστε το αρχείο να διαβάζεται και χωρίς TensorFlow

        models_dir = Path(models_dir)
        with open(models_dir / "intrusion_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_cols = list(meta["num_cols"]) + list(meta["bin_cols"])
        self.cat_cols = list(meta["cat_cols"])
        self.features = list(meta["features"])
        self.preprocessor = joblib.load(models_dir / "intrusion_preprocessor.joblib")
        self.model = keras.models.load_model(models_dir / "intrusion_mlp.keras", compile=False)
        self.threshold = DEFAULT_THRESHOLD

        rule_path = models_dir / "intrusion_rule.json"
        self.rule = None
        if rule_path.exists():
            with open(rule_path, encoding="utf-8") as f:
                self.rule = json.load(f)["attack_paths"]

    def _check_input(self, sessions):
        if isinstance(sessions, dict):
            sessions = pd.DataFrame([sessions])
        if not isinstance(sessions, pd.DataFrame):
            raise TypeError("Η είσοδος πρέπει να είναι pandas DataFrame (ή dict για μία συνεδρία).")
        missing = [c for c in self.features if c not in sessions.columns]
        if missing:
            raise ValueError(f"Λείπουν στήλες εισόδου: {missing}. Απαιτούνται: {self.features}")
        X = sessions[self.features].copy()
        if "encryption_used" in X.columns:
            X["encryption_used"] = X["encryption_used"].where(X["encryption_used"].notna(), "None")
        X[self.num_cols] = X[self.num_cols].apply(pd.to_numeric, errors="coerce")
        bad = X.columns[X.isna().any()].tolist()
        if bad:
            raise ValueError(f"Κενές ή μη αριθμητικές τιμές στις στήλες: {bad}")
        negative = [c for c in self.num_cols if (X[c] < 0).any()]
        if negative:
            raise ValueError(f"Αρνητικές τιμές στις στήλες: {negative} (όλες οι αριθμητικές είσοδοι είναι μη αρνητικές).")
        for c in self.cat_cols:
            X[c] = X[c].astype(str)
        return X

    def _apply_rule(self, X):
        """Ο κανόνας είναι «Ή» από διαδρομές· κάθε διαδρομή είναι «ΚΑΙ» από όρους στις αρχικές μονάδες."""
        ops = {">": np.greater, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}
        fired = np.zeros(len(X), dtype=bool)
        reason = np.full(len(X), "", dtype=object)
        for path in self.rule:
            hit = np.ones(len(X), dtype=bool)
            for cond in path:
                col = X[cond["feature"]].to_numpy()
                hit &= ops[cond["op"]](col, cond["value"])
            text = " και ".join(f"{c['feature']} {c['op']} {c['value']:.4g}" if isinstance(c["value"], float)
                               else f"{c['feature']} {c['op']} {c['value']}" for c in path)
            reason = np.where(hit & ~fired, text, reason)
            fired |= hit
        return fired.astype(int), reason

    def predict(self, sessions, threshold=None):
        """Προβλέψεις για ένα DataFrame συνεδριών. Επιστρέφει DataFrame με το ίδιο index."""
        threshold = self.threshold if threshold is None else float(threshold)
        X = self._check_input(sessions)
        if len(X) == 0:
            return pd.DataFrame(columns=["p_attack", "is_attack", "rule_attack", "rule_reason"])
        p = np.asarray(self.model.predict(self.preprocessor.transform(X).astype("float32"),
                                          batch_size=1024, verbose=0), dtype="float64").ravel()
        out = pd.DataFrame({"p_attack": p, "is_attack": (p >= threshold).astype(int)}, index=X.index)
        if self.rule is not None:
            out["rule_attack"], out["rule_reason"] = self._apply_rule(X)
        return out


def main():
    parser = argparse.ArgumentParser(description="Σύστημα Β: προβλέψεις για συνεδρίες από αρχείο CSV.")
    parser.add_argument("input_csv", help="CSV με τις 9 στήλες εισόδου (επιπλέον στήλες αγνοούνται)")
    parser.add_argument("-o", "--output", help="CSV εξόδου (προεπιλογή: <είσοδος>_predictions.csv)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--threshold", type=float, default=None, help="κατώφλι επίθεσης (προεπιλογή 0.5)")
    args = parser.parse_args()

    sessions = pd.read_csv(args.input_csv, keep_default_na=False, na_values=[""])
    detector = IntrusionDetector(args.models_dir)
    results = detector.predict(sessions, threshold=args.threshold)
    output = args.output or str(Path(args.input_csv).with_name(Path(args.input_csv).stem + "_predictions.csv"))
    pd.concat([sessions, results], axis=1).to_csv(output, index=False)

    print(f"Συνεδρίες: {len(results)} | επιθέσεις κατά το δίκτυο: {int(results['is_attack'].sum())} ({results['is_attack'].mean():.1%})")
    if "rule_attack" in results:
        print(f"Επιθέσεις κατά τον απλό κανόνα: {int(results['rule_attack'].sum())} | "
              f"ίδια απόφαση δικτύου και κανόνα: {(results['is_attack'] == results['rule_attack']).mean():.1%}")
    print("Αποθηκεύτηκε:", output)


if __name__ == "__main__":
    main()
