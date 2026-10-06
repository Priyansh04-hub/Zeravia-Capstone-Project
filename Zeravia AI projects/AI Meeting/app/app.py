import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.extractor import MeetingActionExtractor
from src.validator import validate_items


st.set_page_config(
    page_title="AI Meeting Action-Item Extractor",
    page_icon="📋",
    layout="wide"
)


@st.cache_resource
def load_extractor():
    return MeetingActionExtractor()


st.title("AI Meeting Action-Item Extractor")

st.write(
    "Convert meeting transcripts into structured action items "
    "with owners, deadlines, status and confidence."
)


st.sidebar.header("Transcript Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload meeting transcript",
    type=["txt", "csv"]
)

transcript = ""

if uploaded_file is not None:

    if uploaded_file.name.lower().endswith(".txt"):

        transcript = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

    elif uploaded_file.name.lower().endswith(".csv"):

        uploaded_data = pd.read_csv(uploaded_file)

        transcript_columns = [
            "transcript",
            "Transcript",
            "text",
            "Text",
            "meeting",
            "Meeting",
            "content",
            "Content"
        ]

        selected_column = None

        for column in transcript_columns:

            if column in uploaded_data.columns:
                selected_column = column
                break

        if selected_column is not None:

            transcript = "\n".join(
                uploaded_data[selected_column]
                .astype(str)
            )

        else:

            st.error(
                "No transcript column was found in the CSV."
            )

            st.write(
                list(uploaded_data.columns)
            )


st.subheader("Meeting Transcript")

manual_transcript = st.text_area(
    "Enter or paste transcript",
    value=transcript,
    height=300,
    placeholder=(
        "Example:\n"
        "Manager: Rahul, please prepare the report by Friday.\n"
        "Rahul: I will prepare it.\n"
        "Priya: I will review the report on Monday."
    )
)


if st.button(
    "Extract Action Items",
    type="primary"
):

    if not manual_transcript.strip():

        st.warning(
            "Please enter or upload a meeting transcript."
        )

    else:

        extractor = load_extractor()

        with st.spinner(
            "Analyzing meeting transcript..."
        ):

            extracted = extractor.extract(
                manual_transcript
            )

            validated = validate_items(
                extracted
            )

        if not validated:

            st.error(
                "No action items were detected."
            )

        else:

            st.success(
                f"{len(validated)} action items detected."
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Action Items",
                    len(validated)
                )

            with col2:

                owners = sum(
                    1
                    for item in validated
                    if item["owner"]
                )

                st.metric(
                    "Assigned Owners",
                    owners
                )

            with col3:

                deadlines = sum(
                    1
                    for item in validated
                    if item["deadline"]
                )

                st.metric(
                    "Deadlines Found",
                    deadlines
                )

            st.subheader(
                "Extracted Action Items"
            )

            table_data = []

            for item in validated:

                table_data.append({
                    "Task": item["task"],
                    "Owner": (
                        item["owner"]
                        if item["owner"]
                        else "Not specified"
                    ),
                    "Deadline": (
                        item["deadline"]
                        if item["deadline"]
                        else "Not specified"
                    ),
                    "Status": item["status"],
                    "Confidence": (
                        f"{item['confidence']:.0%}"
                    ),
                    "Warnings": (
                        ", ".join(item["warnings"])
                        if item["warnings"]
                        else "None"
                    )
                })

            result_df = pd.DataFrame(
                table_data
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "Supporting Evidence"
            )

            for index, item in enumerate(
                validated,
                start=1
            ):

                st.markdown(
                    f"**{index}. {item['task']}**"
                )

                st.write(
                    f"Owner: {item['owner'] or 'Not specified'}"
                )

                st.write(
                    f"Deadline: {item['deadline'] or 'Not specified'}"
                )

                st.write(
                    f"Confidence: {item['confidence']:.0%}"
                )

                if item["evidence"]:

                    st.info(
                        item["evidence"]
                    )

                if item["warnings"]:

                    st.warning(
                        ", ".join(item["warnings"])
                    )

            st.divider()

            st.subheader(
                "Structured JSON"
            )

            st.json(
                validated
            )

            csv_data = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download Action Items CSV",
                data=csv_data,
                file_name="action_items.csv",
                mime="text/csv"
            )