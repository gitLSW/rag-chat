import joblib

class DocPathClassifier:
    def __init__(self, company_id):
        self.company_id = company_id
        self.vectorizer = joblib.load(f'./{company_id}/classifier/vectorizer.pkl')
        self.classifier = joblib.load('./{company_id}/classifier/classifier.pkl')

    def classify_doc(self, content):
        # Vectorize the input and predict.
        doc_vector = self.vectorizer.transform([content])
        prediction = self.classifier.predict(doc_vector)
        return prediction[0]