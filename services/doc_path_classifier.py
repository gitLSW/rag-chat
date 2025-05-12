import os
import joblib
from fastapi import HTTPException

class DocPathClassifier:
    def __init__(self, company_id):
        self.company_id = company_id

        base_dir = os.path.dirname(os.path.abspath(__file__))
        classifier_dir = os.path.join(base_dir, '..', 'companies', str(company_id), 'classifier')

        try:
            self.vectorizer = joblib.load(classifier_dir + '/vectorizer.pkl')
            self.classifier = joblib.load(classifier_dir + '/classifier.pkl')
        except FileNotFoundError:
            raise HTTPException(404, f'No path classifier for company {company_id} found. Ask the provider to train you one.')

    def classify_doc(self, content):
        # Vectorize the input and predict.
        doc_vector = self.vectorizer.transform([content])
        prediction = self.classifier.predict(doc_vector)
        return prediction[0]