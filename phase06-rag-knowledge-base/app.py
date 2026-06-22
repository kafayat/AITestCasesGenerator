# Import json for parsing AI output
import json

# Import os for file/folder operations
import os

# Import Streamlit for browser UI
import streamlit as st

# Import Ollama LLM connector
from langchain_ollama import OllamaLLM

# Import Pydantic validation error
from pydantic import ValidationError

# Import Requirement Analyzer Agent
from agents.requirement_analyzer_agent import analyze_requirement

# Import Test Case Generator Agent
from agents.test_case_generator_agent import generate_test_cases

# Import Reviewer Agent
from agents.reviewer_agent import review_test_cases

# Import Pydantic model
from models.test_case_model import TestCaseCollection

# Import Excel writer utility
from utils.excel_writer import save_test_cases_to_excel

# Import RAG knowledge search
from utils.rag_loader import search_knowledge_base


# Create outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)


# Configure Streamlit page
st.set_page_config(
    page_title="AI Test Case Generator with RAG",
    page_icon="🧪",
    layout="wide"
)


# Page title
st.title("AI Test Case Generator - Phase 6 RAG Knowledge Base")


# Page description
st.write(
    "Generate test cases using Multi-Agent Architecture, RAG Knowledge Base, JSON validation, and Excel export."
)


# Sidebar configuration
st.sidebar.header("Configuration")


# Model selector
selected_model = st.sidebar.selectbox(
    "Select AI Model",
    [
        "llama3.2:3b",
        "qwen2.5-coder:7b"
    ]
)


# Create LLM object
llm = OllamaLLM(
    model=selected_model
)


# ==========================================
# Requirement Input Section
# ==========================================

# Allow user to upload a requirement text file
uploaded_file = st.file_uploader(
    "Upload Requirement File",
    type=["txt"]
)


# Default empty requirement
requirement = ""


# If user uploads a file
if uploaded_file is not None:

    # Read file content
    requirement = uploaded_file.read().decode(
        "utf-8"
    )

    # Display uploaded content
    st.subheader("Uploaded Requirement")
    st.text(requirement)


# If no file uploaded, allow manual requirement entry
else:

    requirement = st.text_area(
        "Enter Requirement",
        placeholder="""
Paste your requirement here.

Example:

As a user,
I want to login using username and password,
So that I can access my account.

Acceptance Criteria:
- Valid credentials allow login
- Invalid credentials show error
- Password field is masked
        """,
        height=250
    )


# Generate button
if st.button("Generate Test Cases with RAG"):

    # Check if requirement exists
    if not requirement.strip():

        st.warning(
            "Please upload a requirement file or enter a requirement."
        )

    else:

        # Step 1: Search knowledge base
        with st.spinner("Searching knowledge base..."):

            rag_context = search_knowledge_base(
                requirement
            )

        # Show retrieved RAG context
        st.subheader("Retrieved Knowledge Context")
        st.text(rag_context)


        # Step 2: Analyze requirement
        with st.spinner("Analyzing requirement..."):

            analysis = analyze_requirement(
                llm,
                requirement
            )

        # Show requirement analysis
        st.subheader("Requirement Analysis")
        st.text(analysis)


        # Step 3: Generate test cases using RAG context
        with st.spinner("Generating RAG-based test cases..."):

            raw_test_cases = generate_test_cases(
                llm,
                requirement,
                analysis,
                rag_context
            )


        # Try to validate AI output
        try:

            # Convert AI text response into JSON
            test_case_data = json.loads(
                raw_test_cases
            )

            # Validate JSON structure using Pydantic
            validated_test_cases = TestCaseCollection(
                **test_case_data
            )

            # Convert validated model into formatted JSON
            final_json = validated_test_cases.model_dump_json(
                indent=4
            )

            # Save JSON file
            json_path = "outputs/test_cases.json"

            with open(
                json_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(final_json)


            # Save Excel file
            excel_path = "outputs/test_cases.xlsx"

            save_test_cases_to_excel(
                validated_test_cases.test_cases,
                excel_path
            )


            # Show success message
            st.success(
                "RAG-based test cases generated successfully."
            )


            # Display JSON output
            st.subheader("Generated JSON")
            st.json(test_case_data)


            # Step 4: Reviewer Agent
            with st.spinner("Reviewing test cases..."):

                review = review_test_cases(
                    llm,
                    final_json
                )


            # Show reviewer feedback
            st.subheader("Reviewer Feedback")
            st.write(review)


            # Save analysis output
            with open(
                "outputs/analysis.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(analysis)


            # Save review output
            with open(
                "outputs/review.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(review)


            # Download JSON button
            st.download_button(
                label="Download JSON",
                data=final_json,
                file_name="test_cases.json",
                mime="application/json"
            )


            # Read Excel file for download
            with open(
                excel_path,
                "rb"
            ) as file:

                excel_data = file.read()


            # Download Excel button
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="test_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


        # Handle invalid JSON
        except json.JSONDecodeError:

            st.error("AI returned invalid JSON.")

            st.subheader("Raw AI Response")

            st.text(raw_test_cases)


        # Handle validation error
        except ValidationError as e:

            st.error("JSON validation failed.")

            st.text(str(e))