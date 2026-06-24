import json
import os
import time

import streamlit as st

from pydantic import ValidationError

from agents.requirement_analyzer_agent import analyze_requirement
from agents.test_case_generator_agent import generate_test_cases
from agents.reviewer_agent import review_test_cases
from agents.quality_scoring_agent import score_test_case_quality
from agents.requirement_inference_agent import infer_requirement_from_knowledge

from models.test_case_model import TestCaseCollection

from utils.file_handler import extract_text_from_file
from utils.rag_loader import (
    search_advanced_knowledge_base,
    format_rag_context,
    group_chunks_by_category,
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


def read_requirement_documents(
    requirement_folder="knowledge_base/requirements"
):

    requirement_texts = []

    if not os.path.exists(requirement_folder):
        return ""

    for file_name in os.listdir(requirement_folder):

        if file_name.endswith(".txt"):

            file_path = os.path.join(
                requirement_folder,
                file_name
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

            if content.strip():

                requirement_texts.append(
                    f"Source: {file_name}\n{content}"
                )

    return "\n\n---\n\n".join(requirement_texts)


def get_knowledge_file_count(category):

    category_folder = os.path.join(
        "knowledge_base",
        category
    )

    if not os.path.exists(category_folder):
        return 0

    return len(
        [
            file_name for file_name in os.listdir(category_folder)
            if file_name.endswith(".txt")
        ]
    )


def run_test_case_generation(
    llm,
    requirement,
    analysis,
    rag_context
):

    coverage_instruction = """

Generate maximum practical coverage from the first attempt.
Do not wait for retry to expand coverage.
Do not stop after one scenario.
Do not use a fixed number of test cases.
Use all relevant acceptance criteria, business rules, validations, risks, and retrieved knowledge.
Create separate test cases for each meaningful behaviour.
Cover positive, negative, validation, boundary, business rule, duplicate, failed transaction, confirmation, receipt/reference, security, data integrity, and error handling scenarios where applicable.
Return valid JSON only.
"""

    raw_test_cases = generate_test_cases(
        llm,
        (requirement + coverage_instruction)[:2500],
        analysis[:1500],
        rag_context[:2200]
    )

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

    return raw_test_cases, test_case_data, validated_test_cases


def retry_for_broader_coverage(
    llm,
    requirement,
    analysis,
    rag_context
):

    retry_requirement = requirement + """

The previous output did not provide enough coverage.

Generate broader enterprise coverage.
Do not stop after one scenario.
Do not use a fixed number of test cases.

Create separate test cases for:
- Successful flow
- Invalid input
- Missing mandatory data
- Business rule validation
- Boundary condition
- Duplicate processing
- Failed transaction
- Confirmation notification
- Receipt/reference generation
- Security or sensitive data handling
- Data integrity
- Error handling

Return valid JSON only.
"""

    raw_test_cases = generate_test_cases(
        llm,
        retry_requirement[:2500],
        analysis[:1500],
        rag_context[:2200]
    )

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

    return raw_test_cases, test_case_data, validated_test_cases


st.set_page_config(
    page_title="Enterprise AI Test Case Generator",
    layout="wide"
)


st.title("Enterprise AI Test Case Generator - Phase 11")
st.write(
    "Single upload flow with auto document detection, requirement inference, first-pass maximum coverage, Advanced RAG, Quality Scoring, Excel export, and Word export."
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

run_quality_score = st.sidebar.checkbox(
    "Run Quality Scoring Agent",
    value=True
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


st.header("1. Upload Project Documents")

st.write(
    "Upload any project documents together. The system will automatically detect whether each file is a requirement, business rule, sample test case, standard, defect, or API document."
)

project_files = st.file_uploader(
    "Upload project documents",
    type=allowed_file_types,
    accept_multiple_files=True
)

col1, col2 = st.columns(2)

with col1:

    build_button = st.button(
        "Auto Detect Documents / Build Knowledge Base"
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

    if "detected_requirement" in st.session_state:
        del st.session_state["detected_requirement"]

    if "requirement_source" in st.session_state:
        del st.session_state["requirement_source"]

    st.success(
        "Knowledge base and vector store cleared successfully."
    )


if build_button:

    if project_files:

        classification_results = []
        detected_requirement_texts = []

        for file in project_files:

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

            if detected_category == "requirements":

                detected_requirement_texts.append(
                    f"Source: {file.name}\n{extracted_text}"
                )

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

        combined_requirement = "\n\n---\n\n".join(
            detected_requirement_texts
        )

        if combined_requirement.strip():

            st.session_state["detected_requirement"] = combined_requirement
            st.session_state["requirement_source"] = "Detected from uploaded requirement document"

        else:

            st.session_state["detected_requirement"] = ""
            st.session_state["requirement_source"] = "No requirement document detected"

        st.success(
            "Documents detected, classified, saved, and knowledge base rebuilt successfully."
        )

        st.subheader("Detected Document Types")

        for result in classification_results:

            st.write(
                f"{result['file']} → {result['label']} "
                f"(confidence score: {result['confidence']})"
            )

            with st.expander(
                f"View detection scores for {result['file']}"
            ):
                st.json(result["scores"])

    else:

        st.warning(
            "Please upload at least one project document."
        )


st.subheader("Current Knowledge Base Structure")

for category, label in KNOWLEDGE_CATEGORIES.items():

    file_count = get_knowledge_file_count(category)

    st.write(
        f"{label}: {file_count} file(s)"
    )


st.header("2. Requirement Detection")

requirement = st.session_state.get(
    "detected_requirement",
    ""
)

if not requirement.strip():

    requirement = read_requirement_documents()

if requirement.strip():

    st.success(
        "Requirement found from uploaded documents."
    )

    st.write(
        st.session_state.get(
            "requirement_source",
            "Detected from knowledge base"
        )
    )

    with st.expander("View requirement used for generation"):

        st.text(
            requirement[:3000]
        )

else:

    st.warning(
        "No formal requirement document found. The system can infer a requirement from uploaded knowledge documents."
    )


st.header("3. Generate Test Cases and Quality Score")

if st.button("Generate Enterprise Test Cases"):

    start_time = time.time()

    rag_start_time = time.time()

    with st.spinner("Retrieving categorized enterprise knowledge..."):

        query_text = requirement

        if not query_text.strip():
            query_text = "Generate requirement and test cases from uploaded business rules, sample test cases, standards, defects, and API documentation."

        retrieved_chunks = search_advanced_knowledge_base(
            query_text,
            top_k_per_category=top_k_per_category
        )

        rag_context = format_rag_context(
            retrieved_chunks,
            max_chars=2200
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


    if not requirement.strip():

        with st.spinner("No requirement found. Inferring requirement from uploaded knowledge..."):

            requirement = infer_requirement_from_knowledge(
                llm,
                rag_context
            )

        st.info(
            "Requirement was inferred from uploaded knowledge documents."
        )

        st.subheader("Inferred Requirement")

        st.text(
            requirement
        )

        st.session_state["detected_requirement"] = requirement
        st.session_state["requirement_source"] = "Inferred from uploaded knowledge documents"


    if not requirement.strip():

        st.warning(
            "Unable to generate test cases because no requirement or usable knowledge was found."
        )

        st.stop()


    if fast_mode:

        st.info(
            "Fast Mode enabled: using lightweight requirement analysis."
        )

        analysis = f"""
Feature and Requirement Summary:
{requirement[:1200]}

Relevant Enterprise Knowledge:
{rag_context[:1200]}
"""

    else:

        with st.spinner("Analyzing requirement..."):

            analysis = analyze_requirement(
                llm,
                requirement[:1500]
            )

    st.subheader("Requirement Analysis")

    st.text(
        analysis
    )


    generation_start_time = time.time()

    with st.spinner("Generating enterprise test cases with maximum first-pass coverage..."):

        raw_test_cases, test_case_data, validated_test_cases = run_test_case_generation(
            llm,
            requirement,
            analysis,
            rag_context
        )

    if len(validated_test_cases.test_cases) <= 1:

        st.warning(
            "Only one test case was generated. Retrying automatically for broader coverage..."
        )

        with st.spinner("Retrying generation for maximum coverage..."):

            raw_test_cases, test_case_data, validated_test_cases = retry_for_broader_coverage(
                llm,
                requirement,
                analysis,
                rag_context
            )

    generation_elapsed_time = round(
        time.time() - generation_start_time,
        2
    )

    st.write(
        f"Test case generation completed in {generation_elapsed_time} seconds."
    )


    try:

        final_json = validated_test_cases.model_dump_json(
            indent=4
        )

        with open(
            "outputs/test_cases.json",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                final_json
            )

        quality_report = ""

        if run_quality_score:

            with st.spinner("Scoring generated test case quality..."):

                quality_report = score_test_case_quality(
                    llm,
                    requirement,
                    analysis,
                    rag_context,
                    final_json
                )

            st.subheader("Quality Scoring Report")

            st.write(
                quality_report
            )

            with open(
                "outputs/quality_report.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    quality_report
                )


        export_start_time = time.time()

        output_files = save_outputs(
            validated_test_cases.test_cases,
            quality_report,
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

            st.json(
                test_case_data
            )


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

            st.write(
                review
            )

            with open(
                "outputs/review.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    review
                )


        with open(
            "outputs/analysis.txt",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                analysis
            )


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

        st.error(
            "AI returned invalid JSON even after repair."
        )

        st.subheader(
            "Raw AI Response"
        )

        st.text(
            raw_test_cases
        )

        st.subheader(
            "JSON Error"
        )

        st.text(
            str(e)
        )


    except ValidationError as e:

        st.error(
            "JSON validation failed."
        )

        st.subheader(
            "Raw AI Response"
        )

        st.text(
            raw_test_cases
        )

        st.subheader(
            "Validation Error"
        )

        st.text(
            str(e)
        )