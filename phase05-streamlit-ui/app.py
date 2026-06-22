# Import json library for JSON parsing
import json

# Import os for file and folder handling
import os

# Import Streamlit for web UI
import streamlit as st

# Import Ollama LLM connector
from langchain_ollama import OllamaLLM

# Import Pydantic validation error
from pydantic import ValidationError

# Import agents
from agents.requirement_analyzer_agent import analyze_requirement
from agents.test_case_generator_agent import generate_test_cases
from agents.reviewer_agent import review_test_cases

# Import Pydantic model
from models.test_case_model import TestCaseCollection

# Import Excel writer utility
from utils.excel_writer import save_test_cases_to_excel


# Create outputs folder if it does not already exist
os.makedirs("outputs", exist_ok=True)


# Streamlit page configuration
st.set_page_config(
    page_title="AI Test Case Generator",
    page_icon="🧪",
    layout="wide"
)


# Page title
st.title("AI Test Case Generator - Phase 5")

# Short description
st.write(
    "Generate structured JSON and Excel test cases from business requirements using AI agents."
)


# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection dropdown
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


# Requirement input area
requirement = st.text_area(
    "Enter Requirement",
    value="REQ001: User should be able to login using valid username and password.",
    height=150
)


# Button to generate test cases
if st.button("Generate Test Cases"):

    # Check if requirement is empty
    if not requirement.strip():

        # Show warning if no requirement entered
        st.warning("Please enter a requirement first.")

    else:

        # Step 1: Analyze requirement
        with st.spinner("Analyzing requirement..."):

            analysis = analyze_requirement(
                llm,
                requirement
            )

        # Show analysis
        st.subheader("Requirement Analysis")
        st.text(analysis)


        # Step 2: Generate test cases
        with st.spinner("Generating test cases..."):

            raw_test_cases = generate_test_cases(
                llm,
                requirement,
                analysis
            )


        # Try JSON validation
        try:

            # Convert AI response string into JSON
            test_case_data = json.loads(raw_test_cases)

            # Validate JSON using Pydantic model
            validated_test_cases = TestCaseCollection(
                **test_case_data
            )

            # Convert validated output into formatted JSON
            final_json = validated_test_cases.model_dump_json(
                indent=4
            )

            # Save JSON output
            json_path = "outputs/test_cases.json"

            with open(
                json_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(final_json)

            # Save Excel output
            excel_path = "outputs/test_cases.xlsx"

            save_test_cases_to_excel(
                validated_test_cases.test_cases,
                excel_path
            )

            # Show success message
            st.success("Test cases generated successfully.")

            # Show JSON output on screen
            st.subheader("Generated JSON")
            st.json(test_case_data)


            # Step 3: Review test cases
            with st.spinner("Reviewing test cases..."):

                review = review_test_cases(
                    llm,
                    final_json
                )

            # Show review
            st.subheader("Reviewer Feedback")
            st.write(review)


            # Save analysis and review
            with open(
                "outputs/analysis.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(analysis)

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


        # Handle invalid JSON output
        except json.JSONDecodeError:

            st.error("AI returned invalid JSON.")

            st.subheader("Raw AI Response")

            st.text(raw_test_cases)


        # Handle Pydantic validation errors
        except ValidationError as e:

            st.error("JSON validation failed.")

            st.text(str(e))