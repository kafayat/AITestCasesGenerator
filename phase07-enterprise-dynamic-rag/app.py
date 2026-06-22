import json
import os
import streamlit as st

from langchain_ollama import OllamaLLM
from pydantic import ValidationError

from agents.requirement_analyzer_agent import analyze_requirement
from agents.test_case_generator_agent import generate_test_cases
from agents.reviewer_agent import review_test_cases

from models.test_case_model import TestCaseCollection
from utils.excel_writer import save_test_cases_to_excel
from utils.file_handler import save_uploaded_file, read_txt_file
from utils.rag_loader import search_knowledge_base, reset_vector_store


os.makedirs("outputs", exist_ok=True)
os.makedirs("knowledge_base", exist_ok=True)
os.makedirs("uploaded_knowledge", exist_ok=True)


st.set_page_config(
    page_title="Enterprise AI Test Case Generator",
    page_icon="🧪",
    layout="wide"
)


st.title("Enterprise AI Test Case Generator - Phase 7")
st.write(
    "Dynamic RAG version with uploadable enterprise knowledge, requirement upload, JSON validation, Excel export, and reviewer agent."
)


st.sidebar.header("Configuration")

selected_model = st.sidebar.selectbox(
    "Select AI Model",
    [
        "llama3.2:3b",
        "qwen2.5-coder:7b"
    ]
)

top_k = st.sidebar.slider(
    "Number of RAG documents to retrieve",
    min_value=1,
    max_value=10,
    value=5
)

llm = OllamaLLM(
    model=selected_model
)


st.header("1. Upload Enterprise Knowledge")

knowledge_files = st.file_uploader(
    "Upload QA standards, business rules, sample test cases, BRDs or project rules",
    type=["txt"],
    accept_multiple_files=True
)

if st.button("Save / Rebuild Knowledge Base"):

    if knowledge_files:

        reset_vector_store()

        for file in knowledge_files:
            save_uploaded_file(
                file,
                "knowledge_base"
            )

        st.success("Knowledge files uploaded and vector store reset successfully.")

    else:
        st.warning("Please upload at least one knowledge file.")


st.header("2. Requirement Input")

requirement_file = st.file_uploader(
    "Upload Requirement File",
    type=["txt"]
)

requirement = ""

if requirement_file is not None:
    requirement = read_txt_file(requirement_file)

    st.subheader("Uploaded Requirement")
    st.text(requirement)

else:
    requirement = st.text_area(
        "Or paste requirement manually",
        placeholder="Paste Jira story, user story, acceptance criteria or BRD requirement here...",
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
                requirement,
                analysis,
                rag_context
            )


        try:

            test_case_data = json.loads(
                raw_test_cases
            )

            validated_test_cases = TestCaseCollection(
                **test_case_data
            )

            final_json = validated_test_cases.model_dump_json(
                indent=4
            )

            json_path = "outputs/test_cases.json"

            with open(
                json_path,
                "w",
                encoding="utf-8"
            ) as file:
                file.write(final_json)

            excel_path = "outputs/test_cases.xlsx"

            save_test_cases_to_excel(
                validated_test_cases.test_cases,
                excel_path
            )

            st.success("Enterprise test cases generated successfully.")

            st.subheader("Generated JSON")
            st.json(test_case_data)


            with st.spinner("Reviewing test cases..."):

                review = review_test_cases(
                    llm,
                    final_json,
                    rag_context
                )

            st.subheader("Reviewer Feedback")
            st.write(review)


            with open("outputs/analysis.txt", "w", encoding="utf-8") as file:
                file.write(analysis)

            with open("outputs/review.txt", "w", encoding="utf-8") as file:
                file.write(review)

            st.download_button(
                label="Download JSON",
                data=final_json,
                file_name="test_cases.json",
                mime="application/json"
            )

            with open(excel_path, "rb") as file:
                excel_data = file.read()

            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="test_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


        except json.JSONDecodeError:

            st.error("AI returned invalid JSON.")
            st.subheader("Raw AI Response")
            st.text(raw_test_cases)


        except ValidationError as e:

            st.error("JSON validation failed.")
            st.text(str(e))