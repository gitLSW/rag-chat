import os
import joblib
from fastapi import HTTPException
from utils import get_company_path

class DocPathClassifier:
    def __init__(self, company_id):
        self.company_id = company_id

        classifier_dir = get_company_path(company_id, 'classifier')

        try:
            self.vectorizer = joblib.load(os.path.join(classifier_dir, 'vectorizer.pkl'))
            self.classifier = joblib.load(os.path.join(classifier_dir, 'classifier.pkl'))
        except FileNotFoundError:
            raise HTTPException(404, f"No path classifier for company {company_id} found. Ask the provider to train you one.")

    def classify_doc(self, content):
        # Vectorize the input and predict.
        doc_vector = self.vectorizer.transform([content])
        prediction = self.classifier.predict(doc_vector)
        return prediction[0]