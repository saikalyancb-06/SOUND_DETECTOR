# ==========================================
# FILE: src/classifier_models.py
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import os
import joblib

class EnsembleSpeechClassifier:
    def __init__(self):
        # Task A Classifier: Random Forest Profile Engine
        self.age_model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Task B Classifier: High Performance Gradient Booster
        self.gender_model = XGBClassifier(n_estimators=100, random_state=42)
        # Task C Classifier: Structural Speech Anomaly Evaluator
        self.diagnostic_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)

    def train_pipelines(self, X_train, y_age, y_gender, y_diag):
        """Trains models simultaneously using computed matrix vectors."""
        self.age_model.fit(X_train, y_age)
        self.gender_model.fit(X_train, y_gender)
        self.diagnostic_model.fit(X_train, y_diag)

    def save_pipelines(self, model_dir):
        """Persist trained estimators as reusable artifacts."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.age_model, os.path.join(model_dir, "age_model.joblib"))
        joblib.dump(self.gender_model, os.path.join(model_dir, "gender_model.joblib"))
        joblib.dump(self.diagnostic_model, os.path.join(model_dir, "diagnostic_model.joblib"))

    def load_pipelines(self, model_dir):
        """Load previously trained estimator artifacts if available."""
        self.age_model = joblib.load(os.path.join(model_dir, "age_model.joblib"))
        self.gender_model = joblib.load(os.path.join(model_dir, "gender_model.joblib"))
        self.diagnostic_model = joblib.load(os.path.join(model_dir, "diagnostic_model.joblib"))

    def _fallback_rule_based_prediction(self, feature_vector):
        """Rule-based fallback for demo execution before supervised training artifacts are loaded."""
        mean_f0 = feature_vector.get("mean_f0", 0.0)
        std_f0 = feature_vector.get("std_f0", 0.0)
        rms_energy = feature_vector.get("rms_energy", 0.0)
        mean_zcr = feature_vector.get("mean_zcr", 0.0)

        age_pred = "Child" if mean_f0 >= 220 else "Adult"
        gender_pred = "Female" if mean_f0 >= 165 else "Male"
        atypical_flag = (std_f0 > 120) or (mean_zcr > 0.12) or (rms_energy < 0.01)
        diag_pred = "Atypical" if atypical_flag else "Typical"

        return {
            "Age_Classification": age_pred,
            "Gender_Classification": gender_pred,
            "Acoustic_Profile": diag_pred,
            "Prediction_Mode": "rule_based_fallback"
        }

    def predict_diagnostics(self, feature_vector):
        """Evaluates structural metrics across classification bounds."""
        X = np.array(list(feature_vector.values())).reshape(1, -1)

        try:
            check_is_fitted(self.age_model)
            check_is_fitted(self.gender_model)
            check_is_fitted(self.diagnostic_model)
        except NotFittedError:
            return self._fallback_rule_based_prediction(feature_vector)

        age_pred = "Adult" if self.age_model.predict(X)[0] == 1 else "Child"
        gender_pred = "Male" if self.gender_model.predict(X)[0] == 1 else "Female"
        diag_pred = "Typical" if self.diagnostic_model.predict(X)[0] == 1 else "Atypical"
        
        return {
            "Age_Classification": age_pred,
            "Gender_Classification": gender_pred,
            "Acoustic_Profile": diag_pred,
            "Prediction_Mode": "trained_ensemble"
        }