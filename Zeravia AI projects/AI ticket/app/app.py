import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent
)

sys.path.insert(
    0,
    str(BASE_DIR)
)

from src.predictor import TicketPredictor


st.set_page_config(
    page_title="AI-TicketX",
    page_icon="🎫",
    layout="wide"
)


@st.cache_resource
def load_predictor():

    return TicketPredictor()


predictor = load_predictor()


st.title(
    "AI-TicketX"
)

st.subheader(
    "AI Customer Support Ticket Triage System"
)

st.sidebar.title(
    "Ticket Triage"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload historical ticket CSV",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(
        uploaded_file
    )

    st.sidebar.success(
        f"{len(data):,} tickets loaded"
    )

    st.header(
        "Ticket Analytics"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Total Tickets",
            f"{len(data):,}"
        )

    with col2:

        category_count = (
            data["category"].nunique()
            if "category" in data.columns
            else 0
        )

        st.metric(
            "Categories",
            category_count
        )

    with col3:

        urgency_count = (
            data["urgency"].nunique()
            if "urgency" in data.columns
            else 0
        )

        st.metric(
            "Urgency Levels",
            urgency_count
        )

    with col4:

        st.metric(
            "Model",
            "TF-IDF + Logistic Regression"
        )

    if "category" in data.columns:

        st.subheader(
            "Ticket Categories"
        )

        category_counts = (
            data["category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Tickets"
        ]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Tickets",
            title="Support Tickets by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "urgency" in data.columns:

        st.subheader(
            "Ticket Urgency"
        )

        urgency_counts = (
            data["urgency"]
            .value_counts()
            .reset_index()
        )

        urgency_counts.columns = [
            "Urgency",
            "Tickets"
        ]

        fig = px.bar(
            urgency_counts,
            x="Urgency",
            y="Tickets",
            title="Support Tickets by Urgency"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.divider()

st.header(
    "New Support Ticket"
)

ticket_text = st.text_area(
    "Enter customer support ticket",
    height=180,
    placeholder="Describe the customer's issue..."
)

if st.button(
    "Classify and Route Ticket",
    type="primary"
):

    if not ticket_text.strip():

        st.warning(
            "Please enter a support ticket."
        )

    else:

        result = predictor.predict(
            ticket_text
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Category",
                result["category"].upper()
            )

        with col2:

            st.metric(
                "Urgency",
                result["urgency"].upper()
            )

        with col3:

            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}"
            )

        with col4:

            st.metric(
                "Support Queue",
                result["queue"]
            )

        st.subheader(
            "Triage Result"
        )

        result_data = pd.DataFrame({
            "Field": [
                "Predicted Category",
                "Category Confidence",
                "Predicted Urgency",
                "Urgency Confidence",
                "Overall Confidence",
                "Assigned Queue"
            ],
            "Result": [
                result["category"],
                f"{result['category_confidence']:.2%}",
                result["urgency"],
                f"{result['urgency_confidence']:.2%}",
                f"{result['confidence']:.2%}",
                result["queue"]
            ]
        })

        st.dataframe(
            result_data,
            use_container_width=True,
            hide_index=True
        )

        if result["human_review"]:

            st.error(
                "LOW CONFIDENCE: Ticket has been sent for human review."
            )

        else:

            st.success(
                "HIGH CONFIDENCE: Ticket can be automatically routed."
            )

st.divider()

st.header(
    "Human Review Queue"
)

review_path = (
    BASE_DIR
    / "review_logs"
    / "human_review.csv"
)

if review_path.exists():

    review_data = pd.read_csv(
        review_path
    )

    st.metric(
        "Tickets Awaiting Review",
        len(review_data)
    )

    st.dataframe(
        review_data,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No low-confidence tickets require human review."
    )

st.divider()

st.header(
    "Historical Tickets"
)

if uploaded_file is not None:

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True
    )