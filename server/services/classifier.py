import os
import joblib
import asyncio
import logging
from utils import get_company_path, safe_async_read
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

class Classifier:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier
        

    def classify(self, content):
        # Vectorize the input and predict.
        doc_vector = self.vectorizer.transform([content])
        prediction = self.classifier.predict(doc_vector)
        return prediction[0]
    

    @property
    def num_classes(self):
        return self.classifier.classes_
    

    @classmethod
    def load_from_dir(cls, company_id, plk_dir_name):
        vectorizer_path, classifier_path = Classifier._get_model_paths(company_id, plk_dir_name)

        try:
            vectorizer = joblib.load(vectorizer_path)
            classifier = joblib.load(classifier_path)
            return Classifier(vectorizer, classifier)
        except FileNotFoundError:
            plk_dir_path = os.path.dirname(vectorizer_path)
            raise FileNotFoundError(f"No classifier files found at path {plk_dir_path} found.")


    @classmethod
    async def train(cls, company_id, plk_dir_name, docs_coll, num_classes=None):
        """
        Trains the classifier using documents from the MongoDB collection.
        """
        logger.info(f"Started classifier training for company {company_id}")

        vectorizer_path, classifier_path = Classifier._get_model_paths(company_id, plk_dir_name)

        MAX_LIMIT = 5_000
        limit = num_classes * 500 if num_classes else MAX_LIMIT
        if MAX_LIMIT < limit:
            limit = MAX_LIMIT

        # Fetch data from MongoDB and read corresponding files
        async def prepare_doc_training_data(doc_id, doc_type):
            txt_path = get_company_path(company_id, f'docs/{doc_id}.txt')
            
            # Use the synchronous wrapper for safe_async_read
            try:
                content = await safe_async_read(txt_path)
            except Exception:
                return None

            if content:
                return (content, doc_type)
            else:
                return None
                
        
        prepare_doc_tasks = []
        random_docs_cursor = docs_coll.aggregate([{ "$sample": { "size": limit } }])
        async for doc in random_docs_cursor:
            doc_id = doc.get('id') # Use .get for safety
            doc_type = doc.get('docType')

            if not doc_type:
                continue
            
            prepare_doc_tasks.append(prepare_doc_training_data(doc_id, doc_type))

        texts = []
        labels = []
        for complete_task in asyncio.as_completed(prepare_doc_tasks):
            doc_training_data = await complete_task
            if doc_training_data:
                texts.append(doc_training_data[0])
                labels.append(doc_training_data[1])

        if not texts or not labels or len(texts) != len(labels):
            logger.info(f"No documents found or read in docs of {company_id}. Training cannot proceed.")
            return

        if len(set(labels)) < 2:
            logger.info(f"Not enough class diversity to train in docs of {company_id}. Found labels: {set(labels)}. Need at least 2.")
            return

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None
        )

        # Vectorize the text contents using TF-IDF
        vectorizer = TfidfVectorizer() # Initialize a new vectorizer
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Initialize and train the RandomForest classifier
        classifier = RandomForestClassifier(random_state=42) # Initialize a new classifier
        classifier.fit(X_train_vec, y_train)

        # Evaluate classifier on the testing set
        y_pred = classifier.predict(X_test_vec)
        
        try:
            report = classification_report(y_test, y_pred, zero_division=0)
            logger.info(f"\nClassification Report for company {company_id}:")
            logger.info(report)
        except Exception:
            pass

        # Persist the classifier and the vectorizer to disk
        plk_dir_path = os.path.dirname(vectorizer_path)
        os.makedirs(plk_dir_path, exist_ok=True) # Ensure directory exists
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(classifier, classifier_path)
        
        logger.info(f"Model and vectorizer have been saved to '{vectorizer_path}' and '{classifier_path}'.")
        logger.info("Training complete.")

        return Classifier(vectorizer, classifier)
    

    @staticmethod
    def _get_model_paths(company_id, plk_dir_name):
        plk_dir_path = get_company_path(company_id, plk_dir_name)
        return os.path.join(plk_dir_path, 'vectorizer.pkl'),  os.path.join(plk_dir_path, 'classifier.pkl')