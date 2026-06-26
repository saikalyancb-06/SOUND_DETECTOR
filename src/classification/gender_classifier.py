class GenderClassifier:
    """
    Modular interface wrapper for speaker Gender prediction.
    """
    def __init__(self, classifier_ensemble):
        self.ensemble = classifier_ensemble
        
    def predict(self, feature_vector):
        res = self.ensemble.predict(feature_vector)
        return {
            "gender": res["Gender_Classification"],
            "confidence": res.get("Gender_Confidence", 1.0)
        }
