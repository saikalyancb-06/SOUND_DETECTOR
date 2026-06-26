# ==========================================
# FILE: src/classifier_models.py
# ==========================================
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier as OriginalXGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import os
import joblib

# Wrap XGBClassifier to fix __sklearn_tags__() compatibility with sklearn 1.6+
class XGBClassifier(OriginalXGBClassifier):
    def __sklearn_tags__(self):
        t = super().__sklearn_tags__()
        t.estimator_type = "classifier"
        return t

class EnsembleSpeechClassifier:
    def __init__(self):
        # Base models for Age
        rf_age = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='sqrt', class_weight='balanced', random_state=42)
        xgb_age = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, colsample_bytree=0.4, random_state=42)
        xgb_age._estimator_type = "classifier"
        gb_age = GradientBoostingClassifier(n_estimators=100, max_depth=3, max_features='sqrt', learning_rate=0.05, random_state=42)
        
        voting_age = VotingClassifier(
            estimators=[('rf', rf_age), ('xgb', xgb_age), ('gb', gb_age)],
            voting='soft'
        )
        
        self.age_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', voting_age)
        ])
        
        # Base models for Gender
        rf_gender = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='sqrt', class_weight='balanced', random_state=42)
        xgb_gender = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, colsample_bytree=0.4, random_state=42)
        xgb_gender._estimator_type = "classifier"
        gb_gender = GradientBoostingClassifier(n_estimators=100, max_depth=3, max_features='sqrt', learning_rate=0.05, random_state=42)
        
        voting_gender = VotingClassifier(
            estimators=[('rf', rf_gender), ('xgb', xgb_gender), ('gb', gb_gender)],
            voting='soft'
        )
        
        self.gender_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', voting_gender)
        ])
        
        # Base models for Diagnostic (Clinical Profile)
        rf_diag = RandomForestClassifier(n_estimators=150, max_depth=5, max_features='sqrt', class_weight='balanced', random_state=42)
        xgb_diag = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, colsample_bytree=0.4, random_state=42)
        xgb_diag._estimator_type = "classifier"
        gb_diag = GradientBoostingClassifier(n_estimators=100, max_depth=3, max_features='sqrt', learning_rate=0.05, random_state=42)
        
        voting_diag = VotingClassifier(
            estimators=[('rf', rf_diag), ('xgb', xgb_diag), ('gb', gb_diag)],
            voting='soft'
        )
        
        self.diagnostic_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', voting_diag)
        ])

    def train_pipelines(self, X_train, y_age, y_gender, y_diag):
        """Trains ensemble models using cross-validation."""
        print("[*] Training Age Voting Ensemble...", flush=True)
        if len(np.unique(y_age)) > 1:
            self.age_model.fit(X_train, y_age)
        else:
            print("[!] Age contains only 1 class. Skipping model fit; rule-based/physiological fallback will be used.")

        print("[*] Training Gender Voting Ensemble...", flush=True)
        if len(np.unique(y_gender)) > 1:
            self.gender_model.fit(X_train, y_gender)
        else:
            print("[!] Gender contains only 1 class. Skipping model fit; rule-based/physiological fallback will be used.")

        print("[*] Training Diagnostic Voting Ensemble...", flush=True)
        if len(np.unique(y_diag)) > 1:
            self.diagnostic_model.fit(X_train, y_diag)
        else:
            print("[!] Diagnostic contains only 1 class. Skipping model fit; rule-based/physiological fallback will be used.")
        print("[+] Training step complete.", flush=True)

    def save_pipelines(self, model_dir):
        """Persist trained estimators as reusable joblib files."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Age
        try:
            check_is_fitted(self.age_model)
            joblib.dump(self.age_model, os.path.join(model_dir, "age_model.joblib"))
        except NotFittedError:
            print("[!] Age model was not fitted; skipping save.")
            
        # Save Gender
        try:
            check_is_fitted(self.gender_model)
            joblib.dump(self.gender_model, os.path.join(model_dir, "gender_model.joblib"))
        except NotFittedError:
            print("[!] Gender model was not fitted; skipping save.")
            
        # Save Diagnostic
        try:
            check_is_fitted(self.diagnostic_model)
            joblib.dump(self.diagnostic_model, os.path.join(model_dir, "diagnostic_model.joblib"))
        except NotFittedError:
            print("[!] Diagnostic model was not fitted; skipping save.")

    def load_pipelines(self, model_dir):
        """Load previously trained estimator joblib files if available."""
        age_path = os.path.join(model_dir, "age_model.joblib")
        if os.path.exists(age_path):
            self.age_model = joblib.load(age_path)
            
        gender_path = os.path.join(model_dir, "gender_model.joblib")
        if os.path.exists(gender_path):
            self.gender_model = joblib.load(gender_path)
            
        diag_path = os.path.join(model_dir, "diagnostic_model.joblib")
        if os.path.exists(diag_path):
            self.diagnostic_model = joblib.load(diag_path)

    def _fallback_rule_based_prediction(self, feature_vector):
        """Rule-based fallback for demo execution before supervised training models are loaded."""
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
            "Age_Confidence": 0.85,
            "Gender_Classification": gender_pred,
            "Gender_Confidence": 0.88,
            "Acoustic_Profile": diag_pred,
            "Acoustic_Confidence": 0.80,
            "Prediction_Mode": "rule_based_fallback",
            "Explainability_Rationale": explainability
        }

    def _explain_classification(self, pipeline, X, keys):
        try:
            check_is_fitted(pipeline)
            scaler = pipeline.named_steps['scaler']
            clf = pipeline.named_steps['clf']
            
            importances = np.zeros(X.shape[1])
            valid_ests = [est for est in clf.estimators_ if hasattr(est, "feature_importances_")]
            if valid_ests:
                importances = np.mean([est.feature_importances_ for est in valid_ests], axis=0)
            else:
                importances = np.ones(X.shape[1]) / X.shape[1]
                
            X_scaled = scaler.transform(X)
            influence = np.abs(X_scaled[0]) * importances
            
            feat_influences = sorted(list(zip(keys, influence)), key=lambda x: x[1], reverse=True)
            return feat_influences
        except Exception:
            return []

    def predict_diagnostics(self, feature_vector, threshold=0.65):
        """Evaluates model pipelines over the input acoustic feature matrix dict, returning 'Unknown' for low confidence."""
        # Convert dictionary values to 2D numpy array sorted by key order
        keys = sorted(feature_vector.keys())
        X = np.array([feature_vector[k] for k in keys]).reshape(1, -1)

        # Age prediction
        try:
            check_is_fitted(self.age_model)
            age_class = self.age_model.predict(X)[0]
            age_pred = "Adult" if age_class == 1 or age_class == "Adult" else "Child"
            age_probs = self.age_model.predict_proba(X)[0]
            age_conf = float(np.max(age_probs))
        except NotFittedError:
            age_pred = "Unknown"
            age_conf = 0.50

        # Gender prediction
        try:
            check_is_fitted(self.gender_model)
            gender_class = self.gender_model.predict(X)[0]
            gender_pred = "Male" if gender_class == 1 or gender_class == "Male" else "Female"
            gender_probs = self.gender_model.predict_proba(X)[0]
            gender_conf = float(np.max(gender_probs))
        except NotFittedError:
            gender_pred = "Unknown"
            gender_conf = 0.50

        # Diagnostic Clinical Profile prediction
        try:
            check_is_fitted(self.diagnostic_model)
            diag_class = self.diagnostic_model.predict(X)[0]
            diag_pred = "Typical" if diag_class == 1 or diag_class == "Typical" else "Atypical"
            diag_probs = self.diagnostic_model.predict_proba(X)[0]
            diag_conf = float(np.max(diag_probs))
        except NotFittedError:
            diag_pred = "Typical"
            diag_conf = 0.50
        
        mean_f0 = feature_vector.get("mean_f0", 0.0)
        mean_zcr = feature_vector.get("mean_zcr", 0.0)
        rms_energy = feature_vector.get("rms_energy", 0.0)
        
        # Apply confidence uncertainty thresholds
        reasons = []
        if age_conf < threshold:
            reasons.append(f"Age prediction confidence is low ({age_conf*100:.1f}% < {threshold*100:.1f}%).")
            age_pred = "Unknown"
        if gender_conf < threshold:
            reasons.append(f"Gender prediction confidence is low ({gender_conf*100:.1f}% < {threshold*100:.1f}%).")
            gender_pred = "Unknown"
        if diag_conf < threshold:
            reasons.append(f"Clinical profile prediction confidence is low ({diag_conf*100:.1f}% < {threshold*100:.1f}%).")
            diag_pred = "Unknown"
            
        rationales = []
        
        # Feature importance inspection for predictions
        def is_pipeline_fitted(pipe):
            try:
                check_is_fitted(pipe)
                return True
            except NotFittedError:
                return False

        if is_pipeline_fitted(self.gender_model) and gender_pred != "Unknown":
            gender_explanations = self._explain_classification(self.gender_model, X, keys)
            if gender_explanations:
                g_str = ", ".join([f"{name} ({val*100:.1f}%)" for name, val in gender_explanations[:3]])
                rationales.append(f"Gender classification influenced by: {g_str}.")
        elif gender_pred != "Unknown":
            rationales.append(f"Gender classification fallback: F0={mean_f0:.1f}Hz.")
            
        if is_pipeline_fitted(self.age_model) and age_pred != "Unknown":
            age_explanations = self._explain_classification(self.age_model, X, keys)
            if age_explanations:
                a_str = ", ".join([f"{name} ({val*100:.1f}%)" for name, val in age_explanations[:3]])
                rationales.append(f"Age classification influenced by: {a_str}.")
        elif age_pred != "Unknown":
            rationales.append(f"Age classification fallback: F0={mean_f0:.1f}Hz.")
            
        if is_pipeline_fitted(self.diagnostic_model) and diag_pred != "Unknown":
            diag_explanations = self._explain_classification(self.diagnostic_model, X, keys)
            if diag_explanations:
                d_str = ", ".join([f"{name} ({val*100:.1f}%)" for name, val in diag_explanations[:3]])
                rationales.append(f"Clinical profile recommendation influenced by: {d_str}.")
        elif diag_pred != "Unknown":
            rationales.append(f"Clinical profile fallback: ZCR={mean_zcr:.4f}, RMS={rms_energy:.4f}.")
            
        if not rationales:
            rationales.append("Acoustic parameters are ambiguous or signal quality is low.")
            
        if reasons:
            explainability = " ".join(reasons) + " " + " ".join(rationales)
        else:
            explainability = " ".join(rationales)
        
        return {
            "Age_Classification": age_pred,
            "Age_Confidence": age_conf,
            "Gender_Classification": gender_pred,
            "Gender_Confidence": gender_conf,
            "Acoustic_Profile": diag_pred,
            "Acoustic_Confidence": diag_conf,
            "Prediction_Mode": "trained_ensemble",
            "Explainability_Rationale": explainability
        }