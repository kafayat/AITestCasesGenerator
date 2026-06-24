import json
import os
import time

import streamlit as st

from pydantic import ValidationError

from agents.requirement_analyzer_agent import analyze_requirement
from agents.test_case_generator_agent import generate_test_cases
from agents.reviewer_agent import review_test_cases

from models.test_case_model import TestCaseCollection

from utils.file_handler import extract_text_from_file
from utils.rag_loader import (
    search_advanced_knowledge_base,
    format_rag_context,
    group_chunks_by_category,
    reset_vector_store,
    clear_knowledge_base,
    build_knowledge_base,
    ensure_knowledge_folders,
    KNOWLEDGE_CATEGORIES
)
from utils.json_repair import parse_json_safely, normalize_test_cases
from utils.output_writer import save_outputs
from utils.model_provider import get_llm
from utils.knowledge_classifier import classify_knowledge_file


os.makedirs("outputs", exist_ok=True)
os.makedirs("knowledge_base", exist_ok=True)
os.makedirs("uploaded_knowledge", exist_ok=True)

ensure_knowledge_folders("knowledge_base")


st.set_page_config(
    page_title="Enterprise AI Test Case Generator",
    layout="wide"
)


st.title("Enterprise AI Test Case Generator - Phase 10")
st.write(
    "Improved Advanced Enterprise RAG with auto classification, clean vector rebuild, smaller RAG context, Excel export, and Word export."
)


st.sidebar.header("Configuration")

provider = st.sidebar.selectbox(
    "Model Provider",
    ["Ollama"]
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "llama3.2:1b",
        "llama3.2:3b",
        "qwen2.5:3b",
        "qwen2.5-coder:7b"
    ]
)

fast_mode = st.sidebar.checkbox(
    "Fast Mode",
    value=True
)

top_k_per_category = st.sidebar.slider(
    "Documents per knowledge category",
    min_value=1,
    max_value=2,
    value=1
)

run_reviewer = st.sidebar.checkbox(
    "Run Reviewer Agent",
    value=False
)

show_json = st.sidebar.checkbox(
    "Show JSON on screen",
    value=False
)

show_rag_context = st.sidebar.checkbox(
    "Show Retrieved RAG Context",
    value=True
)

llm = get_llm(
    provider,
    selected_model
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

st.write(
    "Upload knowledge files once. The system automatically classifies them into requirements, business rules, sample test cases, defects, standards, or API docs."
)

knowledge_files = st.file_uploader(
    "Upload enterprise knowledge files",
    type=allowed_file_types,
    accept_multiple_files=True
)

col1, col2 = st.columns(2)

with col1:

    classify_button = st.button(
        "Auto Classify / Save Knowledge / Rebuild Vector Store"
    )

with col2:

    clear_button = st.button(
        "Clear Knowledge Base"
    )


if clear_button:

    clear_knowledge_base(
        "knowledge_base",
        "vector_store"
    )

    st.success(
        "Knowledge base and vector store cleared successfully."
    )


if classify_button:

    if knowledge_files:

        classification_results = []

        for file in knowledge_files:

            extracted_text = extract_text_from_file(file)

            detected_category, confidence_score, scores = classify_knowledge_file(
                file.name,
                extracted_text
            )

            category_folder = os.path.join(
                "knowledge_base",
                detected_category
            )

            os.makedirs(
                category_folder,
                exist_ok=True
            )

            knowledge_file_name = os.path.splitext(file.name)[0] + ".txt"

            knowledge_file_path = os.path.join(
                category_folder,
                knowledge_file_name
            )

            with open(
                knowledge_file_path,
                "w",
                encoding="utf-8"
            ) as output_file:

                output_file.write(extracted_text)

            classification_results.append({
                "file": file.name,
                "category": detected_category,
                "label": KNOWLEDGE_CATEGORIES[detected_category],
                "confidence": confidence_score,
                "scores": scores,
                "path": knowledge_file_path
            })

        build_knowledge_base(
            "knowledge_base",
            "vector_store"
        )

        st.success(
            "Knowledge files classified, saved, and vector store rebuilt successfully."
        )

        st.subheader("Classification Results")

        for result in classification_results:

            st.write(
                f"{result['file']} → {result['label']} "
                f"(confidence score: {result['confidence']})"
            )

            with st.expander(
                f"View classification scores for {result['file']}"
            ):
                st.json(result["scores"])

    else:

        st.warning("Please upload at least one knowledge file.")


st.subheader("Current Knowledge Base Structure")

for category, label in KNOWLEDGE_CATEGORIES.items():

    category_folder = os.path.join(
        "knowledge_base",
        category
    )

    file_count = len(
        [
            file_name for file_name in os.listdir(category_folder)
            if file_name.endswith(".txt")
        ]
    )

    st.write(
        f"{label}: {file_count} file(s)"
    )


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

        start_time = time.time()

        rag_start_time = time.time()

        with st.spinner("Retrieving categorized enterprise knowledge..."):

            retrieved_chunks = search_advanced_knowledge_base(
                requirement,
                top_k_per_category=top_k_per_category
            )

            rag_context = format_rag_context(
                retrieved_chunks,
                max_chars=900
            )

        rag_elapsed_time = round(
            time.time() - rag_start_time,
            2
        )

        st.write(
            f"RAG retrieval completed in {rag_elapsed_time} seconds."
        )

        if show_rag_context:

            st.subheader("Advanced Retrieved RAG Context")

            grouped_chunks = group_chunks_by_category(
                retrieved_chunks
            )

            for category_label, chunks in grouped_chunks.items():

                with st.expander(
                    f"{category_label} Retrieved ({len(chunks)})"
                ):

                    for chunk in chunks:

                        st.write(
                            f"Source: {chunk['source']} | Chunk: {chunk['chunk_number']}"
                        )

                        st.text(
                            chunk["content"][:800]
                        )


        if fast_mode:

            st.info(
                "Fast Mode enabled: using lightweight requirement analysis."
            )

            analysis = f"""
Feature and Requirement Summary:
{requirement[:600]}

Relevant Enterprise Knowledge:
{rag_context[:700]}
"""

        else:

            with st.spinner("Analyzing requirement..."):

                analysis = analyze_requirement(
                    llm,
                    requirement[:1000]
                )

        st.subheader("Requirement Analysis")
        st.text(analysis)


        generation_start_time = time.time()

        with st.spinner("Generating enterprise test cases..."):

            raw_test_cases = generate_test_cases(
                llm,
                requirement[:900],
                analysis[:700],
                rag_context[:900]
            )

        generation_elapsed_time = round(
            time.time() - generation_start_time,
            2
        )

        st.write(
            f"Test case generation completed in {generation_elapsed_time} seconds."
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

            export_start_time = time.time()

            output_files = save_outputs(
                validated_test_cases.test_cases,
                "outputs"
            )

            export_elapsed_time = round(
                time.time() - export_start_time,
                2
            )

            st.write(
                f"Export completed in {export_elapsed_time} seconds."
            )

            elapsed_time = round(
                time.time() - start_time,
                2
            )

            st.success(
                f"Enterprise test cases generated successfully in {elapsed_time} seconds."
            )

            if show_json:

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