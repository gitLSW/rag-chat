import joblib

class DocPathClassifier:
    def __init__(self, company_id):
        self.company_id = company_id
        self.vectorizer = joblib.load('./models/vectorizer.pkl')
        self.classifier = joblib.load('./models/classifier.pkl')

    def classify_doc(self, content):
        # Vectorize the input and predict.
        doc_vector = self.vectorizer.transform([content])
        prediction = self.classifier.predict(doc_vector)
        return prediction[0]