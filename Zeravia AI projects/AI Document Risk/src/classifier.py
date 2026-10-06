import joblib
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing import clean_text


class DocumentRiskClassifier:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )

        self.classes_ = None

    def prepare_text(self, texts):

        return [
            clean_text(text)
            for text in texts
        ]

    def fit(self, texts, labels):

        texts = self.prepare_text(texts)

        X = self.vectorizer.fit_transform(
            texts
        )

        self.model.fit(
            X,
            labels
        )

        self.classes_ = self.model.classes_

        return self

    def predict(self, texts):

        texts = self.prepare_text(texts)

        X = self.vectorizer.transform(
            texts
        )

        predictions = self.model.predict(
            X
        )

        probabilities = self.model.predict_proba(
            X
        )

        confidences = probabilities.max(
            axis=1
        )

        return predictions, confidences

    def predict_one(
        self,
        text,
        threshold=0.60
    ):

        prediction, confidence = self.predict(
            [text]
        )

        label = prediction[0]
        confidence = float(
            confidence[0]
        )

        manual_review = (
            confidence < threshold
        )

        return {
            "category": label,
            "confidence": confidence,
            "manual_review": manual_review
        }

    def save(self, model_path, vectorizer_path):

        joblib.dump(
            self.model,
            model_path
        )

        joblib.dump(
            self.vectorizer,
            vectorizer_path
        )

    def load(
        self,
        model_path,
        vectorizer_path
    ):

        self.model = joblib.load(
            model_path
        )

        self.vectorizer = joblib.load(
            vectorizer_path
        )

        self.classes_ = self.model.classes_

        return self