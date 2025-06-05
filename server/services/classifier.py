import os
import joblib
from utils import get_company_path, safe_async_read
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class Classifier:
    def __init__(self, company_id, plk_dir_name):
        self.company_id = company_id

        self.plk_dir_path = get_company_path(company_id, plk_dir_name)
        self.classifier_path = os.path.join(self.plk_dir_path, 'classifier.pkl')
        self.vectorizer_path = os.path.join(self.plk_dir_path, 'vectorizer.pkl')

        try:
            self.vectorizer = joblib.load(self.classifier_path)
            self.classifier = joblib.load(self.vectorizer_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No classifier files found at path {self.plk_dir_path} found.")
        

    def classify_doc(self, content):
        # Vectorize the input and predict.
        doc_vector = self.vectorizer.transform([content])
        prediction = self.classifier.predict(doc_vector)
        return prediction[0]
    
    
    async def train(self, docs_db_coll):
        """
        Trains the classifier using documents from the MongoDB collection.

        Args:
            docs_db_coll: A MongoDB collection object. Each document in the
                          collection should have an 'id' and 'docType' field.
        """
        print(f"Started classifier training for company {self.company_id}")
        texts = []
        labels = []

        # Fetch data from MongoDB and read corresponding files
        # Limiting to 1000 documents for faster example runs if DB is large. Remove .limit() for full DB.
        for doc in docs_db_coll.find(): # Add .limit(1000) for testing on large DBs
            doc_id = doc.get('id') # Use .get for safety
            doc_type = doc.get('docType')

            if not doc_type:
                continue
            
            txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
            
            # Use the synchronous wrapper for safe_async_read
            try:
                content = await safe_async_read(txt_path)
            except Exception:
                continue

            if content:
                texts.append(content)
                labels.append(doc_type)

        if not texts or not labels or len(texts) != len(labels):
            print("No documents found or read. Training cannot proceed.")
            return

        if len(set(labels)) < 2:
            print(f"Not enough class diversity to train. Found labels: {set(labels)}. Need at least 2.")
            return

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None
        )

        # Vectorize the text contents using TF-IDF
        self.vectorizer = TfidfVectorizer() # Initialize a new vectorizer
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Initialize and train the RandomForest classifier
        self.classifier = RandomForestClassifier(random_state=42) # Initialize a new classifier
        self.classifier.fit(X_train_vec, y_train)

        # Evaluate classifier on the testing set
        y_pred = self.classifier.predict(X_test_vec)
        
        try:
            report = classification_report(y_test, y_pred, zero_division=0)
            print("\nClassification Report:")
            print(report)
        except Exception:
            pass

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.2f}")

        # Persist the classifier and the vectorizer to disk
        os.makedirs(self.plk_dir_path, exist_ok=True) # Ensure directory exists
        
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.classifier, self.classifier_path)
        
        print(f"Model and vectorizer have been saved to '{self.vectorizer_path}' and '{self.classifier_path}'.")
        print("Training complete.")