import json
import os
import time

import streamlit as st

from pydantic import ValidationError

from agents.requirement_analyzer_agent import analyze_requirement
from agents.test_case_generator_agent import generate_test_cases_concurrent
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
from utils.json_repair import normalize_test_cases
from utils.output_writer import save_outputs
from utils.model_provider import get_llm
from utils.knowledge_classifier import classify_knowledge_file
from utils.cache import clear_cache


os.makedirs("outputs", exist_ok=True)
os.makedirs("knowledge_base", exist_ok=True)
os.makedirs("uploaded_knowledge", exist_ok=True)
os.makedirs("cache", exist_ok=True)

ensure_knowledge_folders("knowledge_base")


GROUP_LABELS = {
    "happy_path": "Happy Path / Confirmation / Receipt",
    "negative_boundary": "Invalid Input / Missing Data / Boundary",
    "business_rules_integrity": "Business Rules / Duplicate / Data Integrity",
    "failure_security_error": "Failure / Security / Error Handling",
}


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


def stream_text_response(llm, prompt, placeholder):
    """
    Streams tokens from a single-call agent into a Streamlit placeholder
    so the UI shows live progress instead of a blank spinner. Falls back
    to a single blocking invoke() if streaming isn't supported by the
    underlying LLM wrapper.
    """

    full_text = ""

    try:
        for chunk in llm.stream(prompt):
            full_text += chunk
            placeholder.markdown(full_text)
        return full_text

    except Exception:
        full_text = llm.invoke(prompt)
        placeholder.markdown(full_text)
        return full_text


st.set_page_config(
    page_title="Enterprise AI Test Case Generator",
    layout="wide"
)


st.title("Enterprise AI Test Case Generator - Phase 12")
st.write(
    "Concurrent coverage-group generation, response caching, streaming "
    "agent output, and task-based model routing on top of the Phase 11 "
    "single-upload flow."
)


st.sidebar.header("Configuration")

provider = st.sidebar.selectbox(
    "Model Provider",
    ["Ollama"]
)

available_models = [
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen2.5:3b",
    "qwen2.5-coder:7b"
]

generation_model = st.sidebar.selectbox(
    "Generation Model (fast, called concurrently per coverage group)",
    available_models,
    index=0,
)

quality_model = st.sidebar.selectbox(
    "Quality / Review Model (called once, can be larger)",
    available_models,
    index=0,
)

st.sidebar.caption(
    "For real concurrency speedups, run Ollama with "
    "OLLAMA_NUM_PARALLEL >= 4 so it can serve the 4 coverage-group "
    "requests at the same time instead of queueing them."
)

fast_mode = st.sidebar.checkbox(
    "Fast Mode (lightweight requirement analysis, skips analyzer LLM call)",
    value=True
)

use_cache = st.sidebar.checkbox(
    "Use response cache (skip LLM call on identical re-run)",
    value=True
)

if st.sidebar.button("Clear response cache"):
    clear_cache()
    st.sidebar.success("Response cache cleared.")

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
    "Upload any project documents together. The system will automatically "
    "detect whether each file is a requirement, business rule, sample "
    "test case, standard, defect, or API document."
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

        st.subheader("Document Classification Results")

        for result in classification_results:

            st.write(
                f"**{result['file']}** -> {result['label']} "
                f"(confidence score: {result['confidence']})"
            )

    else:

        st.warning(
            "Please upload at least one document before building the knowledge base."
        )


st.header("2. Requirement")

st.subheader("Knowledge Base Status")

status_cols = st.columns(len(KNOWLEDGE_CATEGORIES))

for index, (category, label) in enumerate(KNOWLEDGE_CATEGORIES.items()):

    with status_cols[index]:
        st.metric(label, get_knowledge_file_count(category))

requirement = st.session_state.get("detected_requirement", "")

if not requirement:
    requirement = read_requirement_documents()

if requirement.strip():

    st.success(
        st.session_state.get(
            "requirement_source",
            "Requirement loaded from knowledge base."
        )
    )

    st.text_area(
        "Detected / Active Requirement",
        value=requirement,
        height=200,
        disabled=True
    )

else:

    st.warning(
        "No formal requirement document found. The system can infer a "
        "requirement from uploaded knowledge documents."
    )


st.header("3. Generate Test Cases and Quality Score")

if st.button("Generate Enterprise Test Cases"):

    start_time = time.time()

    generation_llm = get_llm(provider, generation_model, task="generation")
    analysis_llm = get_llm(provider, generation_model, task="analysis")
    inference_llm = get_llm(provider, generation_model, task="inference")
    quality_llm = get_llm(provider, quality_model, task="scoring")
    review_llm = get_llm(provider, quality_model, task="review")

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

        st.subheader("Inferred Requirement")
        inferred_placeholder = st.empty()

        with st.spinner("No requirement found. Inferring requirement from uploaded knowledge..."):

            prompt_template = open(
                "prompts/requirement_inference_prompt.txt",
                "r",
                encoding="utf-8"
            ).read()

            inference_prompt = prompt_template.replace(
                "{rag_context}", rag_context[:2500]
            )

            requirement = stream_text_response(
                inference_llm, inference_prompt, inferred_placeholder
            )

        st.info(
            "Requirement was inferred from uploaded knowledge documents."
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
            "Fast Mode enabled: using lightweight requirement analysis (no extra LLM call)."
        )

        analysis = f"""
Feature and Requirement Summary:
{requirement[:1200]}

Relevant Enterprise Knowledge:
{rag_context[:1200]}
"""

        st.subheader("Requirement Analysis")
        st.text(analysis)

    else:

        st.subheader("Requirement Analysis")
        analysis_placeholder = st.empty()

        with st.spinner("Analyzing requirement..."):

            prompt_template = open(
                "prompts/requirement_analyzer_prompt.txt",
                "r",
                encoding="utf-8"
            ).read()

            analysis_prompt = prompt_template.replace(
                "{requirement}", requirement[:1500]
            )

            analysis = stream_text_response(
                analysis_llm, analysis_prompt, analysis_placeholder
            )

    generation_start_time = time.time()

    st.subheader("Test Case Generation Progress")

    group_status_placeholders = {}
    progress_cols = st.columns(len(GROUP_LABELS))

    for index, (group_id, label) in enumerate(GROUP_LABELS.items()):
        with progress_cols[index]:
            st.caption(label)
            group_status_placeholders[group_id] = st.empty()
            group_status_placeholders[group_id].info("Queued")

    def progress_callback(group_id, status):
        placeholder = group_status_placeholders.get(group_id)
        if not placeholder:
            return
        if status == "started":
            placeholder.warning("Generating...")
        elif status == "done":
            placeholder.success("Done")
        elif status == "error":
            placeholder.error("Repair needed")

    coverage_instruction = (
        "\n\nUse all relevant acceptance criteria, business rules, "
        "validations, and retrieved knowledge. Coverage is more "
        "important than quantity."
    )

    with st.spinner("Generating test cases concurrently across coverage groups..."):

        merged_result, group_results = generate_test_cases_concurrent(
            generation_llm,
            (requirement + coverage_instruction)[:2500],
            analysis[:1500],
            rag_context[:2200],
            model_name=generation_model,
            use_cache=use_cache,
            progress_callback=progress_callback,
        )

    generation_elapsed_time = round(
        time.time() - generation_start_time,
        2
    )

    cache_hits = sum(1 for r in group_results if r["from_cache"])

    st.write(
        f"Test case generation completed in {generation_elapsed_time} "
        f"seconds across {len(group_results)} concurrent groups "
        f"({cache_hits} served from cache)."
    )

    failed_groups = [r for r in group_results if r["error"]]

    if failed_groups:

        with st.expander(f"{len(failed_groups)} group(s) needed repair / had issues"):

            for result in failed_groups:

                st.write(f"**{result['group_id']}**: {result['error']}")
                st.text(result["raw_text"][:1500])

    test_case_data = normalize_test_cases(
        merged_result,
        default_requirement_id="REQ001"
    )

    try:

        validated_test_cases = TestCaseCollection(**test_case_data)

        final_json = validated_test_cases.model_dump_json(indent=4)

        with open(
            "outputs/test_cases.json",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(final_json)

        st.success(
            f"Generated {len(validated_test_cases.test_cases)} test cases "
            f"across {len(GROUP_LABELS)} coverage groups."
        )

        quality_report = ""

        if run_quality_score:

            st.subheader("Quality Scoring Report")
            quality_placeholder = st.empty()

            with st.spinner("Scoring generated test case quality..."):

                prompt_template = open(
                    "prompts/quality_scoring_prompt.txt",
                    "r",
                    encoding="utf-8"
                ).read()

                quality_prompt = (
                    prompt_template
                    .replace("{requirement}", requirement[:1200])
                    .replace("{analysis}", analysis[:800])
                    .replace("{rag_context}", rag_context[:1000])
                    .replace("{test_cases}", final_json[:2500])
                )

                quality_report = stream_text_response(
                    quality_llm, quality_prompt, quality_placeholder
                )

            with open(
                "outputs/quality_report.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(quality_report)

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
            f"Enterprise test cases generated successfully in {elapsed_time} seconds total."
        )

        if show_json:

            st.subheader("Generated JSON")

            st.json(test_case_data)

        if run_reviewer and not fast_mode:

            st.subheader("Reviewer Feedback")
            review_placeholder = st.empty()

            with st.spinner("Reviewing test cases..."):

                prompt_template = open(
                    "prompts/reviewer_prompt.txt",
                    "r",
                    encoding="utf-8"
                ).read()

                review_prompt = (
                    prompt_template
                    .replace("{requirement}", requirement[:1000])
                    .replace("{analysis}", analysis[:800])
                    .replace("{rag_context}", rag_context[:1000])
                    .replace("{test_cases}", final_json[:1500])
                )

                review = stream_text_response(
                    review_llm, review_prompt, review_placeholder
                )

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

        st.error(
            "AI returned invalid JSON even after repair."
        )

        st.subheader("JSON Error")
        st.text(str(e))

    except ValidationError as e:

        st.error(
            "JSON validation failed."
        )

        st.subheader("Validation Error")
        st.text(str(e))
