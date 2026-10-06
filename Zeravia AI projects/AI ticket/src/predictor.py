from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd

from src.preprocessing import clean_text


class TicketPredictor:

    def __init__(self):

        base_dir = (
            Path(__file__)
            .resolve()
            .parent.parent
        )

        self.vectorizer = joblib.load(
            base_dir / "models" / "vectorizer.pkl"
        )

        self.category_model = joblib.load(
            base_dir / "models" / "category_model.pkl"
        )

        self.urgency_model = joblib.load(
            base_dir / "models" / "urgency_model.pkl"
        )

        self.review_dir = (
            base_dir / "review_logs"
        )

        self.review_dir.mkdir(
            exist_ok=True
        )

        self.confidence_threshold = 0.65

    def detect_category(self, text):

        text = text.lower()

        billing_phrases = [
            "charged twice",
            "charged two times",
            "double charge",
            "charged again",
            "wrong charge",
            "incorrect charge",
            "unexpected charge",
            "duplicate charge",
            "refund",
            "refund request",
            "money back",
            "billing",
            "billing issue",
            "billing problem",
            "payment failed",
            "payment problem",
            "payment issue",
            "invoice",
            "subscription payment",
            "cancel subscription"
        ]

        account_phrases = [
            "cannot log in",
            "can't log in",
            "cannot login",
            "can't login",
            "unable to login",
            "unable to log in",
            "forgot password",
            "reset password",
            "password reset",
            "account locked",
            "locked out",
            "account access",
            "access my account",
            "sign in",
            "signin"
        ]

        technical_phrases = [
            "application keeps crashing",
            "app keeps crashing",
            "application crashes",
            "app crashes",
            "keeps crashing",
            "crashes when",
            "crash when",
            "application crash",
            "app crash",
            "software crash",
            "not working",
            "doesn't work",
            "does not work",
            "won't work",
            "cannot open",
            "can't open",
            "unable to open",
            "keeps freezing",
            "application freezes",
            "app freezes",
            "technical problem",
            "technical issue",
            "error message",
            "system error",
            "software error",
            "bug"
        ]

        product_phrases = [
            "product information",
            "product details",
            "product features",
            "product feature",
            "information about the product",
            "information about upgrading",
            "information about upgrade",
            "want to upgrade",
            "want to upgrade my",
            "upgrade my plan",
            "upgrade plan",
            "pricing plan",
            "available plans",
            "different plans",
            "features available",
            "product availability",
            "product specifications",
            "product compatibility"
        ]

        billing_score = sum(
            1
            for phrase in billing_phrases
            if phrase in text
        )

        account_score = sum(
            1
            for phrase in account_phrases
            if phrase in text
        )

        technical_score = sum(
            1
            for phrase in technical_phrases
            if phrase in text
        )

        product_score = sum(
            1
            for phrase in product_phrases
            if phrase in text
        )

        scores = {
            "billing": billing_score,
            "account": account_score,
            "technical": technical_score,
            "product": product_score
        }

        best_category = max(
            scores,
            key=scores.get
        )

        best_score = scores[
            best_category
        ]

        if best_score >= 1:

            return (
                best_category,
                0.95
            )

        return None, 0.0

    def predict(self, text):

        cleaned_text = clean_text(
            text
        )

        rule_category, rule_confidence = (
            self.detect_category(
                cleaned_text
            )
        )

        features = self.vectorizer.transform(
            [cleaned_text]
        )

        category_probabilities = (
            self.category_model.predict_proba(
                features
            )[0]
        )

        urgency_probabilities = (
            self.urgency_model.predict_proba(
                features
            )[0]
        )

        category_index = (
            category_probabilities.argmax()
        )

        urgency_index = (
            urgency_probabilities.argmax()
        )

        ml_category = (
            self.category_model.classes_[
                category_index
            ]
        )

        urgency = (
            self.urgency_model.classes_[
                urgency_index
            ]
        )

        ml_category_confidence = float(
            category_probabilities[
                category_index
            ]
        )

        urgency_confidence = float(
            urgency_probabilities[
                urgency_index
            ]
        )

        if rule_category is not None:

            category = rule_category
            category_confidence = rule_confidence

        else:

            category = ml_category
            category_confidence = (
                ml_category_confidence
            )

        overall_confidence = min(
            category_confidence,
            urgency_confidence
        )

        human_review = (
            overall_confidence
            < self.confidence_threshold
        )

        queue = self.get_queue(
            category
        )

        if human_review:

            self.log_review(
                text,
                category,
                urgency,
                overall_confidence,
                queue
            )

        return {
            "category": category,
            "urgency": urgency,
            "category_confidence": category_confidence,
            "urgency_confidence": urgency_confidence,
            "confidence": overall_confidence,
            "human_review": human_review,
            "queue": queue
        }

    def get_queue(self, category):

        queues = {
            "billing": "Billing Support",
            "technical": "Technical Support",
            "account": "Account Support",
            "product": "Product Support"
        }

        return queues.get(
            category,
            "General Support"
        )

    def log_review(
        self,
        text,
        category,
        urgency,
        confidence,
        queue
    ):

        path = (
            self.review_dir
            / "human_review.csv"
        )

        row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "ticket_text": text,
            "predicted_category": category,
            "predicted_urgency": urgency,
            "confidence": confidence,
            "assigned_queue": queue
        }])

        if path.exists():

            row.to_csv(
                path,
                mode="a",
                header=False,
                index=False
            )

        else:

            row.to_csv(
                path,
                index=False
            )