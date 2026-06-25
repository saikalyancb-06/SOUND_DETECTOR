# ==========================================
# FILE: src/classifier_models.py
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import os
import joblib

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, f_classif

class EnsembleSpeechClassifier:
    def __init__(self):
        # Task A Classifier: Random Forest Profile Engine with scaling, selection, and balanced weight
        self.age_model = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=15)),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=4, class_weight='balanced', random_state=42))
        ])
        
        # Task B Classifier: High Performance Gradient Booster with scaling, selection, and L2 regularization
        self.gender_model = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=15)),
            ('clf', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8, random_state=42))
        ])
        
        # Task C Classifier: Structural Speech Anomaly Evaluator with scaling, selection, and balanced weight
        self.diagnostic_model = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=15)),
            ('clf', RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, class_weight='balanced', random_state=42))
        ])

    def train_pipelines(self, X_train, y_age, y_gender, y_diag):
        """Trains models simultaneously using GridSearchCV to find optimal parameters."""
        print("[*] Tuning Age Model via GridSearchCV...", flush=True)
        age_param_grid = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [4, 6, None],
            'clf__min_samples_split': [2, 4]
        }
        grid_age = GridSearchCV(self.age_model, age_param_grid, cv=3, scoring='balanced_accuracy')
        grid_age.fit(X_train, y_age)
        self.age_model = grid_age.best_estimator_
        print(f"[+] Age Model Tuned Params: {grid_age.best_params_}", flush=True)

        print("[*] Tuning Gender Model via GridSearchCV...", flush=True)
        gender_param_grid = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [3, 4],
            'clf__learning_rate': [0.05, 0.1]
        }
        grid_gender = GridSearchCV(self.gender_model, gender_param_grid, cv=3, scoring='balanced_accuracy')
        grid_gender.fit(X_train, y_gender)
        self.gender_model = grid_gender.best_estimator_
        print(f"[+] Gender Model Tuned Params: {grid_gender.best_params_}", flush=True)

        print("[*] Tuning Diagnostic Model via GridSearchCV...", flush=True)
        diag_param_grid = {
            'clf__n_estimators': [100, 150],
            'clf__max_depth': [4, 6],
            'clf__min_samples_split': [2, 4]
        }
        grid_diag = GridSearchCV(self.diagnostic_model, diag_param_grid, cv=3, scoring='balanced_accuracy')
        grid_diag.fit(X_train, y_diag)
        self.diagnostic_model = grid_diag.best_estimator_
        print(f"[+] Diagnostic Model Tuned Params: {grid_diag.best_params_}", flush=True)

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

        explainability = (
            f"Rule-based estimation: F0 = {mean_f0:.1f}Hz implies {gender_pred} {age_pred}. "
            f"Acoustic profile marked as {diag_pred} due to spectral/energy dispersion metrics."
        )

        return {
            "Age_Classification": age_pred,
            "Gender_Classification": gender_pred,
            "Acoustic_Profile": diag_pred,
            "Prediction_Mode": "rule_based_fallback",
            "Explainability_Rationale": explainability
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
        
        # Robust clinical/physical acoustic override: any voiced speaker with F0 < 160 Hz is physiologically an Adult Male
        mean_f0 = feature_vector.get("mean_f0", 0.0)
        if 0.0 < mean_f0 < 160.0:
            gender_pred = "Male"
            age_pred = "Adult"
            
        mean_zcr = feature_vector.get("mean_zcr", 0.0)
        rms_energy = feature_vector.get("rms_energy", 0.0)
        
        rationales = []
        if 0.0 < mean_f0 < 160.0:
            rationales.append(f"Fundamental frequency (F0: {mean_f0:.1f}Hz) resides in typical adult male acoustic bounds (<160Hz).")
        elif mean_f0 >= 220.0:
            rationales.append(f"Fundamental frequency (F0: {mean_f0:.1f}Hz) resides in child/pediatric acoustic bounds (>=220Hz).")
        elif mean_f0 >= 160.0:
            rationales.append(f"Fundamental frequency (F0: {mean_f0:.1f}Hz) resides in adult female acoustic bounds (160-220Hz).")
        else:
            rationales.append("Fundamental frequency (F0) was unvoiced or below minimum tracking thresholds.")
            
        if diag_pred == "Typical":
            rationales.append(f"Spectral temporal energy parameters are stable (ZCR: {mean_zcr:.4f}, RMS Energy: {rms_energy:.4f}), matching typical controls.")
        else:
            rationales.append(f"Spectral temporal energy fluctuations detected (ZCR: {mean_zcr:.4f}, RMS Energy: {rms_energy:.4f}), matching clinical atypical bounds.")
            
        explainability = " ".join(rationales)
        
        return {
            "Age_Classification": age_pred,
            "Gender_Classification": gender_pred,
            "Acoustic_Profile": diag_pred,
            "Prediction_Mode": "trained_ensemble",
            "Explainability_Rationale": explainability
        }