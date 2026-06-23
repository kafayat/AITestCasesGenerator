import json
import os

import streamlit as st

from langchain_ollama import OllamaLLM
from pydantic import ValidationError

from agents.requirement_analyzer_agent import analyze_requirement
from agents.test_case_generator_agent import generate_test_cases
from agents.reviewer_agent import review_test_cases

from models.test_case_model import TestCaseCollection

from utils.file_handler import extract_text_from_file
from utils.rag_loader import search_knowledge_base, reset_vector_store
from utils.json_repair import parse_json_safely, normalize_test_cases
from utils.output_writer import save_outputs


os.makedirs("outputs", exist_ok=True)
os.makedirs("knowledge_base", exist_ok=True)
os.makedirs("uploaded_knowledge", exist_ok=True)


st.set_page_config(
    page_title="Enterprise AI Test Case Generator",
    page_icon="🧪",
    layout="wide"
)


st.title("Enterprise AI Test Case Generator - Phase 8")
st.write(
    "Fast Mode with RAG, prompt engineering, JSON repair, Excel export, and Word document export."
)


st.sidebar.header("Configuration")

selected_model = st.sidebar.selectbox(
    "Select AI Model",
    [
        "llama3.2:3b",
        "qwen2.5-coder:7b"
    ]
)

fast_mode = st.sidebar.checkbox(
    "Fast Mode",
    value=True
)

top_k = st.sidebar.slider(
    "Number of RAG documents to retrieve",
    min_value=1,
    max_value=3,
    value=1
)

run_reviewer = st.sidebar.checkbox(
    "Run Reviewer Agent",
    value=False
)

llm = OllamaLLM(
    model=selected_model
)


allowed_file_types = [
    "txt",
    "pdf",
    "docx",
    "xlsx",
    "csv",
    "json"
]


st.header("1. Upload Enterprise Knowledge")

knowledge_files = st.file_uploader(
    "Upload QA standards, business rules, sample test cases, BRDs, PDFs, DOCX, Excel, CSV or JSON files",
    type=allowed_file_types,
    accept_multiple_files=True
)

if st.button("Save / Rebuild Knowledge Base"):

    if knowledge_files:

        reset_vector_store()

        for file in knowledge_files:

            extracted_text = extract_text_from_file(file)

            knowledge_file_name = os.path.splitext(file.name)[0] + ".txt"

            knowledge_file_path = os.path.join(
                "knowledge_base",
                knowledge_file_name
            )

            with open(
                knowledge_file_path,
                "w",
                encoding="utf-8"
            ) as output_file:

                output_file.write(extracted_text)

        st.success(
            "Knowledge files uploaded, converted to text, and vector store reset successfully."
        )

    else:

        st.warning("Please upload at least one enterprise knowledge file.")


st.header("2. Requirement Input")

requirement_file = st.file_uploader(
    "Upload Requirement File",
    type=allowed_file_types
)

requirement = ""

if requirement_file is not None:

    requirement = extract_text_from_file(requirement_file)

    st.subheader("Uploaded Requirement")
    st.text(requirement)

else:

    requirement = st.text_area(
        "Or paste requirement manually",
        placeholder="Paste Jira story, user story, acceptance criteria, BRD requirement, API rule, or business requirement here...",
        height=220
    )


if st.button("Generate Enterprise Test Cases"):

    if not requirement.strip():

        st.warning("Please upload or enter a requirement.")

    else:

        with st.spinner("Retrieving enterprise knowledge..."):

            rag_context = search_knowledge_base(
                requirement,
                top_k=top_k
            )

        st.subheader("Retrieved RAG Context")
        st.text(rag_context)


        if fast_mode:

            st.info("Fast Mode enabled: skipping detailed Requirement Analyzer Agent.")

            analysis = f"""
Feature and Requirement Summary:
{requirement[:800]}

Relevant RAG Context:
{rag_context[:800]}
"""

        else:

            with st.spinner("Analyzing requirement..."):

                analysis = analyze_requirement(
                    llm,
                    requirement
                )

        st.subheader("Requirement Analysis")
        st.text(analysis)


        with st.spinner("Generating enterprise test cases..."):

            raw_test_cases = generate_test_cases(
                llm,
                requirement[:1200],
                analysis[:1000],
                rag_context[:1000]
            )


        try:

            test_case_data = parse_json_safely(
                raw_test_cases,
                llm
            )

            test_case_data = normalize_test_cases(
                test_case_data,
                default_requirement_id="REQ001"
            )

            validated_test_cases = TestCaseCollection(
                **test_case_data
            )

            final_json = validated_test_cases.model_dump_json(
                indent=4
            )

            with open(
                "outputs/test_cases.json",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(final_json)

            output_files = save_outputs(
                validated_test_cases.test_cases,
                "outputs"
            )

            st.success("Enterprise test cases generated successfully.")

            st.subheader("Generated JSON")
            st.json(test_case_data)


            if run_reviewer and not fast_mode:

                with st.spinner("Reviewing test cases..."):

                    review = review_test_cases(
                        llm,
                        requirement,
                        analysis,
                        rag_context,
                        final_json
                    )

                st.subheader("Reviewer Feedback")
                st.write(review)

                with open(
                    "outputs/review.txt",
                    "w",
                    encoding="utf-8"
                ) as file:

                    file.write(review)


            with open(
                "outputs/analysis.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(analysis)


            with open(output_files["excel"], "rb") as file:
                excel_data = file.read()

            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="test_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            with open(output_files["word"], "rb") as file:
                word_data = file.read()

            st.download_button(
                label="Download Word Document",
                data=word_data,
                file_name="test_cases.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )


        except json.JSONDecodeError as e:

            st.error("AI returned invalid JSON even after repair.")
            st.subheader("Raw AI Response")
            st.text(raw_test_cases)
            st.subheader("JSON Error")
            st.text(str(e))


        except ValidationError as e:

            st.error("JSON validation failed.")
            st.subheader("Raw AI Response")
            st.text(raw_test_cases)
            st.subheader("Validation Error")
            st.text(str(e))