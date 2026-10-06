import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.classifier import DocumentRiskClassifier
from src.preprocessing import clean_text
from src.validator import validate_prediction, get_top_terms


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "document_classifier.pkl"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "vectorizer.pkl"
)


st.set_page_config(
    page_title="AI Document Risk & Compliance Classifier",
    page_icon="📄",
    layout="wide"
)


@st.cache_resource
def load_classifier():

    classifier = DocumentRiskClassifier()

    classifier.load(
        MODEL_PATH,
        VECTORIZER_PATH
    )

    return classifier


st.title(
    "AI Document Risk & Compliance Classifier"
)

st.write(
    "Analyze business and legal document text, "
    "classify its compliance category, "
    "explain the prediction and flag uncertain cases "
    "for manual review."
)


classifier = load_classifier()


st.sidebar.header(
    "Document Input"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload document",
    type=["txt", "csv"]
)


document_text = ""


if uploaded_file is not None:

    if uploaded_file.name.lower().endswith(
        ".txt"
    ):

        document_text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

    elif uploaded_file.name.lower().endswith(
        ".csv"
    ):

        uploaded_data = pd.read_csv(
            uploaded_file
        )

        possible_columns = [
            "text",
            "Text",
            "document",
            "Document",
            "content",
            "Content"
        ]

        selected_column = None

        for column in possible_columns:

            if column in uploaded_data.columns:

                selected_column = column

                break

        if selected_column:

            document_text = "\n".join(
                uploaded_data[
                    selected_column
                ].astype(str)
            )

        else:

            st.error(
                "No document text column found."
            )

            st.write(
                list(
                    uploaded_data.columns
                )
            )


st.subheader(
    "Document Text"
)


document_text = st.text_area(
    "Paste document text",
    value=document_text,
    height=350,
    placeholder=(
        "Paste a contract clause or "
        "business document here..."
    )
)


threshold = st.sidebar.slider(
    "Manual Review Threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.60,
    step=0.05
)


if st.button(
    "Analyze Document",
    type="primary"
):

    if not document_text.strip():

        st.warning(
            "Please upload or enter document text."
        )

    else:

        cleaned_text = clean_text(
            document_text
        )

        result = classifier.predict_one(
            cleaned_text,
            threshold
        )

        validation = validate_prediction(
            result["category"],
            result["confidence"],
            threshold
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Category",
                result["category"]
            )

        with col2:

            st.metric(
                "Confidence",
                f"{result['confidence']:.1%}"
            )

        with col3:

            status = (
                "Manual Review"
                if result["manual_review"]
                else "Automatically Classified"
            )

            st.metric(
                "Decision",
                status
            )

        if result["manual_review"]:

            st.warning(
                "This document has low prediction "
                "confidence and should be reviewed manually."
            )

        else:

            st.success(
                "Prediction confidence is above "
                "the review threshold."
            )

        st.divider()

        st.subheader(
            "Document Classification"
        )

        st.write(
            f"Category: **{result['category']}**"
        )

        st.write(
            f"Confidence: **{result['confidence']:.2%}**"
        )

        st.write(
            f"Review threshold: **{threshold:.2%}**"
        )

        st.divider()

        st.subheader(
            "Important Terms"
        )

        terms = get_top_terms(
            classifier.vectorizer,
            classifier.model,
            cleaned_text,
            top_n=10
        )

        if terms:

            terms_df = pd.DataFrame(
                terms
            )

            st.dataframe(
                terms_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No strong explanatory terms were found."
            )

        st.divider()

        st.subheader(
            "Validation"
        )

        if validation["warnings"]:

            for warning in validation[
                "warnings"
            ]:

                st.warning(
                    warning
                )

        else:

            st.success(
                "No validation warnings."
            )

        st.divider()

        st.subheader(
            "Structured Result"
        )

        st.json({
            "category": result["category"],
            "confidence": result["confidence"],
            "manual_review": result["manual_review"],
            "warnings": validation["warnings"]
        })

        result_df = pd.DataFrame([
            {
                "Category": result["category"],
                "Confidence": result["confidence"],
                "Manual Review": result["manual_review"]
            }
        ])

        csv_data = result_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Analysis",
            data=csv_data,
            file_name="document_risk_analysis.csv",
            mime="text/csv"
        )