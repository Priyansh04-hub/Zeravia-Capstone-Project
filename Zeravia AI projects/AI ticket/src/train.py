from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customer_support_tickets.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


data = pd.read_csv(DATA_PATH)


data = data.dropna(
    subset=[
        "Ticket Subject",
        "Ticket Description",
        "Ticket Type",
        "Ticket Priority"
    ]
)


data["description"] = (
    data["Ticket Description"]
    .astype(str)
    .str.replace(
        "{product_purchased}",
        "product",
        regex=False
    )
)


data["text"] = (
    data["Ticket Subject"].astype(str)
    + " "
    + data["description"]
)


data["text"] = data["text"].apply(
    clean_text
)


def create_category(row):

    text = row["text"].lower()

    ticket_type = (
        str(row["Ticket Type"])
        .lower()
        .strip()
    )

    account_words = [
        "login",
        "log in",
        "logged in",
        "password",
        "account",
        "sign in",
        "signin",
        "access account",
        "locked account",
        "username",
        "authentication"
    ]

    if any(
        word in text
        for word in account_words
    ):
        return "account"

    if ticket_type == "technical issue":
        return "technical"

    if ticket_type in [
        "billing inquiry",
        "refund request",
        "cancellation request"
    ]:
        return "billing"

    if ticket_type == "product inquiry":
        return "product"

    return "technical"


data["category"] = data.apply(
    create_category,
    axis=1
)


priority_mapping = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "high"
}


data["urgency"] = (
    data["Ticket Priority"]
    .astype(str)
    .str.lower()
    .map(priority_mapping)
)


data = data.dropna(
    subset=["urgency"]
)


X = data["text"]

category = data["category"]

urgency = data["urgency"]


X_train, X_test, y_category_train, y_category_test = (
    train_test_split(
        X,
        category,
        test_size=0.2,
        random_state=42,
        stratify=category
    )
)


X_train_urgency, X_test_urgency, y_urgency_train, y_urgency_test = (
    train_test_split(
        X,
        urgency,
        test_size=0.2,
        random_state=42,
        stratify=urgency
    )
)


vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

X_train_urgency_tfidf = vectorizer.transform(
    X_train_urgency
)

X_test_urgency_tfidf = vectorizer.transform(
    X_test_urgency
)


category_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=2.0
)


urgency_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=2.0
)


category_model.fit(
    X_train_tfidf,
    y_category_train
)


urgency_model.fit(
    X_train_urgency_tfidf,
    y_urgency_train
)


category_predictions = category_model.predict(
    X_test_tfidf
)


urgency_predictions = urgency_model.predict(
    X_test_urgency_tfidf
)


category_precision, category_recall, category_f1, _ = (
    precision_recall_fscore_support(
        y_category_test,
        category_predictions,
        average="weighted"
    )
)


urgency_precision, urgency_recall, urgency_f1, _ = (
    precision_recall_fscore_support(
        y_urgency_test,
        urgency_predictions,
        average="weighted"
    )
)


results = pd.DataFrame({
    "Model": [
        "Category Classifier",
        "Urgency Classifier"
    ],
    "Accuracy": [
        accuracy_score(
            y_category_test,
            category_predictions
        ),
        accuracy_score(
            y_urgency_test,
            urgency_predictions
        )
    ],
    "Precision": [
        category_precision,
        urgency_precision
    ],
    "Recall": [
        category_recall,
        urgency_recall
    ],
    "F1": [
        category_f1,
        urgency_f1
    ]
})


print(
    "\nCATEGORY DISTRIBUTION\n"
)

print(
    data["category"].value_counts()
)


print(
    "\nMODEL PERFORMANCE\n"
)

print(
    results.to_string(
        index=False
    )
)


print(
    "\nCATEGORY CLASSIFICATION REPORT\n"
)

print(
    classification_report(
        y_category_test,
        category_predictions
    )
)


print(
    "\nURGENCY CLASSIFICATION REPORT\n"
)

print(
    classification_report(
        y_urgency_test,
        urgency_predictions
    )
)


print(
    "\nCATEGORY CONFUSION MATRIX\n"
)

print(
    confusion_matrix(
        y_category_test,
        category_predictions
    )
)


print(
    "\nURGENCY CONFUSION MATRIX\n"
)

print(
    confusion_matrix(
        y_urgency_test,
        urgency_predictions
    )
)


joblib.dump(
    vectorizer,
    MODELS_DIR / "vectorizer.pkl"
)


joblib.dump(
    category_model,
    MODELS_DIR / "category_model.pkl"
)


joblib.dump(
    urgency_model,
    MODELS_DIR / "urgency_model.pkl"
)


results.to_csv(
    REPORTS_DIR / "model_results.csv",
    index=False
)


print(
    "\nModels saved successfully."
)