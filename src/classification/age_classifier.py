class AgeClassifier:
    """
    Modular interface wrapper for speaker Age prediction.
    """
    def __init__(self, classifier_ensemble):
        self.ensemble = classifier_ensemble
        
    def predict(self, feature_vector):
        res = self.ensemble.predict(feature_vector)
        return {
            "age": res["Age_Classification"],
            "confidence": res.get("Age_Confidence", 1.0)
        }
