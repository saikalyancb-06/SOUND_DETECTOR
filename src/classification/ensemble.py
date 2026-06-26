from src.classifier_models import EnsembleSpeechClassifier

class SpeechClassifierEnsemble:
    """
    Ensemble classifier orchestrator for gender, age, and typicality predictions.
    """
    def __init__(self):
        self.classifier = EnsembleSpeechClassifier()
        
    def train(self, X_train, y_age, y_gender, y_diag):
        self.classifier.train_pipelines(X_train, y_age, y_gender, y_diag)
        
    def save(self, model_dir):
        self.classifier.save_pipelines(model_dir)
        
    def load(self, model_dir):
        self.classifier.load_pipelines(model_dir)
        
    def predict(self, feature_vector):
        return self.classifier.predict_diagnostics(feature_vector)
