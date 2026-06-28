import json
import os
import queue
import re
import threading
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pydantic import ValidationError

from agents.test_case_generator_agent import generate_test_cases_concurrent

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
from utils.coverage_analysis import (
    ALL_COVERAGE_AREAS,
    build_gap_analysis_report,
    format_report_as_markdown,
    format_facts_for_quality_prompt,
)
from utils.quality_checks import annotate_quality_flags


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
    "lifecycle_operations": "Amend / Replace / Certificate Generation",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def read_requirement_documents(requirement_folder="knowledge_base/requirements"):

    requirement_texts = []

    if not os.path.exists(requirement_folder):
        return ""

    for file_name in os.listdir(requirement_folder):

        if file_name.endswith(".txt"):

            file_path = os.path.join(requirement_folder, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            if content.strip():
                requirement_texts.append(f"Source: {file_name}\n{content}")

    return "\n\n---\n\n".join(requirement_texts)


def extract_requirement_id(requirement_text, fallback="REQ001"):
    """
    Looks for a requirement ID pattern like "REQ-SW001" or "REQ001" at
    the start of the requirement text, so all 4 generation groups can be
    forced to agree on one ID instead of each independently guessing.
    """

    match = re.search(r"\bREQ[-_]?[A-Za-z0-9]+\b", requirement_text[:300])

    return match.group(0) if match else fallback


def get_knowledge_file_count(category):

    category_folder = os.path.join("knowledge_base", category)

    if not os.path.exists(category_folder):
        return 0

    return len([f for f in os.listdir(category_folder) if f.endswith(".txt")])


def stream_text_response(llm, prompt, placeholder):
    """
    Streams tokens from a single-call agent into a Streamlit placeholder
    so the UI shows live progress instead of a blank spinner.
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


def status_pill(status_text, is_good):
    """
    Small colored HTML pill for quick visual scanning of Covered/MISSING
    style statuses in dataframes and lists - this is the kind of thing
    that reads as "finished product" rather than "script output" in a
    demo, with zero new dependencies (just inline styled markdown).
    """

    color = "#1a7f37" if is_good else "#b42318"
    background = "#dafbe1" if is_good else "#fde2e1"

    return (
        f'<span style="background-color:{background}; color:{color}; '
        f'padding:2px 10px; border-radius:12px; font-size:0.85em; '
        f'font-weight:600;">{status_text}</span>'
    )


def render_coverage_heatmap(coverage_matrix):
    """
    Renders the coverage matrix as a single-column heatmap (areas on the
    Y axis, test-case count as color intensity) - the single highest
    "wow factor per effort" visual for a stakeholder demo: gaps (count
    0, shown in red) are immediately visible without reading any text.
    """

    areas = [row["coverage_area"] for row in coverage_matrix]
    counts = [row["count"] for row in coverage_matrix]

    figure = go.Figure(
        data=go.Heatmap(
            z=[[count] for count in counts],
            y=areas,
            x=["Test Case Count"],
            colorscale=[[0.0, "#fde2e1"], [0.001, "#fff3cd"], [1.0, "#1a7f37"]],
            showscale=False,
            text=[[count] for count in counts],
            texttemplate="%{text}",
            textfont={"size": 14},
            zmin=0,
        )
    )

    figure.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )

    return figure


def render_dataframe_with_pills(rows, text_col, status_col, ids_col, text_label):
    """
    Builds a pandas DataFrame for st.dataframe with an HTML-pilled status
    column. st.dataframe doesn't render raw HTML in cells, so for the
    colored-pill effect we use st.markdown on a manually built HTML
    table instead - still zero new dependencies, just templated HTML.
    """

    html_rows = []

    for row in rows:

        ids_text = ", ".join(row[ids_col]) if row[ids_col] else "-"
        is_good = row[status_col] in ("Covered", "Done")
        pill = status_pill(row[status_col], is_good)

        html_rows.append(
            f"<tr><td style='padding:6px 10px;'>{row[text_col]}</td>"
            f"<td style='padding:6px 10px;'>{ids_text}</td>"
            f"<td style='padding:6px 10px;'>{pill}</td></tr>"
        )

    table_html = (
        "<table style='width:100%; border-collapse:collapse;'>"
        f"<tr><th style='text-align:left; padding:6px 10px;'>{text_label}</th>"
        "<th style='text-align:left; padding:6px 10px;'>Matched Test Cases</th>"
        "<th style='text-align:left; padding:6px 10px;'>Status</th></tr>"
        + "".join(html_rows)
        + "</table>"
    )

    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Enterprise AI Test Case Generator", layout="wide")

st.title("Enterprise AI Test Case Generator - Phase 14")
st.caption(
    "Concurrent coverage-group generation, deterministic coverage & gap "
    "analysis, and a tabbed dashboard UI on top of the Phase 12/13 engine."
)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

st.sidebar.header("Configuration")

provider = st.sidebar.selectbox("Model Provider", ["Ollama"])

available_models = [
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen2.5:3b",
    "qwen2.5-coder:7b",
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

ui_mode = st.sidebar.radio(
    "UI Mode",
    ["Demo (clean progress bar)", "Developer (live streaming JSON)"],
    index=0,
)
dev_mode = ui_mode.startswith("Developer")

fast_mode = st.sidebar.checkbox(
    "Fast Mode (lightweight requirement analysis, skips analyzer LLM call)",
    value=True,
)

use_cache = st.sidebar.checkbox(
    "Use response cache (skip LLM call on identical re-run)", value=True
)

if st.sidebar.button("Clear response cache"):
    clear_cache()
    st.sidebar.success("Response cache cleared.")

top_k_per_category = st.sidebar.slider(
    "Documents per knowledge category", min_value=1, max_value=2, value=1
)

run_quality_score = st.sidebar.checkbox("Run Quality Scoring Agent", value=True)
run_reviewer = st.sidebar.checkbox("Run Reviewer Agent", value=False)
show_json = st.sidebar.checkbox("Show JSON on screen", value=False)
show_rag_context = st.sidebar.checkbox("Show Retrieved RAG Context", value=True)


allowed_file_types = ["txt", "pdf", "docx", "xlsx", "csv", "json"]


# ---------------------------------------------------------------------------
# Top-of-page summary dashboard - reads from session_state, so it's visible
# on every tab/rerun once a generation has completed, not just right after
# the button click.
# ---------------------------------------------------------------------------

results = st.session_state.get("phase14_results")

if results:

    dashboard_cols = st.columns(5)

    with dashboard_cols[0]:
        st.metric("Test Cases", len(results["test_case_data"]["test_cases"]))

    with dashboard_cols[1]:
        hit = len(ALL_COVERAGE_AREAS) - len(results["gap_report"]["missing_coverage_areas"])
        st.metric("Coverage Areas Hit", f"{hit}/{len(ALL_COVERAGE_AREAS)}")

    with dashboard_cols[2]:
        high_risk_count = len(results["gap_report"]["risk_coverage"].get("High", []))
        st.metric("High-Risk Test Cases", high_risk_count)

    with dashboard_cols[3]:
        st.metric("Generation Time", f"{results['generation_elapsed_time']}s")

    with dashboard_cols[4]:
        st.metric("Total Run Time", f"{results['total_elapsed_time']}s")

    st.divider()


tab_upload, tab_requirement, tab_generate, tab_coverage, tab_quality = st.tabs([
    "1. Upload & Knowledge Base",
    "2. Requirement",
    "3. Generate",
    "4. Coverage & Gap Analysis",
    "5. Quality & Review",
])


# ---------------------------------------------------------------------------
# Tab 1 - Upload & Knowledge Base
# ---------------------------------------------------------------------------

with tab_upload:

    st.write(
        "Upload any project documents together. The system will automatically "
        "detect whether each file is a requirement, business rule, sample "
        "test case, standard, defect, or API document."
    )

    project_files = st.file_uploader(
        "Upload project documents",
        type=allowed_file_types,
        accept_multiple_files=True,
        key="project_files_uploader",
    )

    col1, col2 = st.columns(2)

    with col1:
        build_button = st.button("Auto Detect Documents / Build Knowledge Base")

    with col2:
        clear_button = st.button("Clear Knowledge Base")

    if clear_button:

        clear_knowledge_base("knowledge_base", "vector_store")

        for key in ("detected_requirement", "requirement_source", "phase14_results"):
            st.session_state.pop(key, None)

        st.success("Knowledge base and vector store cleared successfully.")

    if build_button:

        if project_files:

            classification_results = []
            detected_requirement_texts = []

            for file in project_files:

                extracted_text = extract_text_from_file(file)

                detected_category, confidence_score, scores = classify_knowledge_file(
                    file.name, extracted_text
                )

                category_folder = os.path.join("knowledge_base", detected_category)
                os.makedirs(category_folder, exist_ok=True)

                knowledge_file_name = os.path.splitext(file.name)[0] + ".txt"
                knowledge_file_path = os.path.join(category_folder, knowledge_file_name)

                with open(knowledge_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(extracted_text)

                if detected_category == "requirements":
                    detected_requirement_texts.append(f"Source: {file.name}\n{extracted_text}")

                classification_results.append({
                    "file": file.name,
                    "category": detected_category,
                    "label": KNOWLEDGE_CATEGORIES[detected_category],
                    "confidence": confidence_score,
                })

            build_knowledge_base("knowledge_base", "vector_store")

            combined_requirement = "\n\n---\n\n".join(detected_requirement_texts)

            if combined_requirement.strip():
                st.session_state["detected_requirement"] = combined_requirement
                st.session_state["requirement_source"] = "Detected from uploaded requirement document"
            else:
                st.session_state["detected_requirement"] = ""
                st.session_state["requirement_source"] = "No requirement document detected"

            st.success("Documents detected, classified, saved, and knowledge base rebuilt successfully.")

            st.subheader("Document Classification Results")

            classification_df = pd.DataFrame(classification_results).rename(columns={
                "file": "File", "label": "Detected Category", "confidence": "Confidence Score"
            })[["File", "Detected Category", "Confidence Score"]]

            st.dataframe(classification_df, use_container_width=True, hide_index=True)

        else:
            st.warning("Please upload at least one document before building the knowledge base.")

    st.subheader("Knowledge Base Status")

    status_cols = st.columns(len(KNOWLEDGE_CATEGORIES))

    for index, (category, label) in enumerate(KNOWLEDGE_CATEGORIES.items()):
        with status_cols[index]:
            st.metric(label, get_knowledge_file_count(category))


# ---------------------------------------------------------------------------
# Tab 2 - Requirement
# ---------------------------------------------------------------------------

requirement = st.session_state.get("detected_requirement", "")

if not requirement:
    requirement = read_requirement_documents()

with tab_requirement:

    if requirement.strip():

        st.success(st.session_state.get("requirement_source", "Requirement loaded from knowledge base."))

        st.text_area(
            "Detected / Active Requirement",
            value=requirement,
            height=280,
            disabled=True,
        )

    else:

        st.warning(
            "No formal requirement document found. The system can infer a "
            "requirement from uploaded knowledge documents when you generate "
            "test cases."
        )


# ---------------------------------------------------------------------------
# Tab 3 - Generate
# ---------------------------------------------------------------------------

with tab_generate:

    generate_clicked = st.button("Generate Enterprise Test Cases", type="primary")

    if generate_clicked:

        start_time = time.time()

        generation_llm = get_llm(provider, generation_model, task="generation")
        analysis_llm = get_llm(provider, generation_model, task="analysis")
        inference_llm = get_llm(provider, generation_model, task="inference")
        quality_llm = get_llm(provider, quality_model, task="scoring")
        review_llm = get_llm(provider, quality_model, task="review")

        rag_start_time = time.time()

        with st.spinner("Retrieving categorized enterprise knowledge..."):

            query_text = requirement.strip() or (
                "Generate requirement and test cases from uploaded business "
                "rules, sample test cases, standards, defects, and API documentation."
            )

            retrieved_chunks = search_advanced_knowledge_base(
                query_text, top_k_per_category=top_k_per_category
            )

            rag_context = format_rag_context(retrieved_chunks, max_chars=2200)

        rag_elapsed_time = round(time.time() - rag_start_time, 2)
        st.write(f"RAG retrieval completed in {rag_elapsed_time} seconds.")

        if show_rag_context:

            with st.expander("Advanced Retrieved RAG Context"):

                grouped_chunks = group_chunks_by_category(retrieved_chunks)

                for category_label, chunks in grouped_chunks.items():
                    st.markdown(f"**{category_label} Retrieved ({len(chunks)})**")
                    for chunk in chunks:
                        st.caption(f"Source: {chunk['source']} | Chunk: {chunk['chunk_number']}")
                        st.text(chunk["content"][:800])

        if not requirement.strip():

            st.subheader("Inferred Requirement")
            inferred_placeholder = st.empty()

            with st.spinner("No requirement found. Inferring requirement from uploaded knowledge..."):

                prompt_template = open(
                    "prompts/requirement_inference_prompt.txt", "r", encoding="utf-8"
                ).read()

                inference_prompt = prompt_template.replace("{rag_context}", rag_context[:2500])

                requirement = stream_text_response(inference_llm, inference_prompt, inferred_placeholder)

            st.info("Requirement was inferred from uploaded knowledge documents.")

            st.session_state["detected_requirement"] = requirement
            st.session_state["requirement_source"] = "Inferred from uploaded knowledge documents"

        if not requirement.strip():
            st.warning("Unable to generate test cases because no requirement or usable knowledge was found.")
            st.stop()

        if fast_mode:

            st.info("Fast Mode enabled: using lightweight requirement analysis (no extra LLM call).")

            analysis = (
                f"Feature and Requirement Summary:\n{requirement[:1200]}\n\n"
                f"Relevant Enterprise Knowledge:\n{rag_context[:1200]}"
            )

            with st.expander("Requirement Analysis"):
                st.text(analysis)

        else:

            st.subheader("Requirement Analysis")
            analysis_placeholder = st.empty()

            with st.spinner("Analyzing requirement..."):

                prompt_template = open(
                    "prompts/requirement_analyzer_prompt.txt", "r", encoding="utf-8"
                ).read()

                analysis_prompt = prompt_template.replace("{requirement}", requirement[:1500])

                analysis = stream_text_response(analysis_llm, analysis_prompt, analysis_placeholder)

        generation_start_time = time.time()

        st.subheader("Test Case Generation Progress")

        active_requirement_id = extract_requirement_id(requirement)

        stream_queue = queue.Queue()
        generation_outcome = {}

        coverage_instruction = (
            "\n\nUse all relevant acceptance criteria, business rules, "
            "validations, and retrieved knowledge. Coverage is more "
            "important than quantity."
        )

        def _run_generation():
            try:
                result = generate_test_cases_concurrent(
                    generation_llm,
                    (requirement + coverage_instruction)[:2500],
                    analysis[:1500],
                    rag_context[:2200],
                    model_name=generation_model,
                    requirement_id=active_requirement_id,
                    use_cache=use_cache,
                    stream_queue=stream_queue,
                )
                generation_outcome["result"] = result
            except Exception as exc:
                generation_outcome["error"] = str(exc)

        generation_thread = threading.Thread(target=_run_generation, daemon=True)
        generation_thread.start()

        finished_groups = set()

        if dev_mode:

            group_panels = {}
            group_status_labels = {}
            progress_cols = st.columns(len(GROUP_LABELS))

            for index, (group_id, label) in enumerate(GROUP_LABELS.items()):
                with progress_cols[index]:
                    st.caption(label)
                    group_status_labels[group_id] = st.empty()
                    group_status_labels[group_id].info("Queued")
                    group_panels[group_id] = st.empty()

            timer_placeholder = st.empty()

        else:

            demo_progress_bar = st.progress(0, text="Starting generation...")
            demo_caption = st.empty()

        while generation_thread.is_alive() or not stream_queue.empty():

            drained_any = False

            while True:
                try:
                    kind, group_id, payload = stream_queue.get_nowait()
                except queue.Empty:
                    break

                drained_any = True

                if kind == "chunk" and dev_mode:
                    group_status_labels[group_id].warning("Generating...")
                    group_panels[group_id].code(payload[-600:], language="json")

                elif kind == "group_done":
                    finished_groups.add(group_id)
                    if dev_mode:
                        if payload == "done":
                            group_status_labels[group_id].success("Done")
                        elif payload == "truncated":
                            group_status_labels[group_id].warning("Done (possibly truncated)")
                        else:
                            group_status_labels[group_id].error("Repair needed")

            elapsed = round(time.time() - generation_start_time, 1)
            fraction_done = len(finished_groups) / len(GROUP_LABELS)

            if dev_mode:
                timer_placeholder.caption(
                    f"{elapsed}s elapsed - {len(finished_groups)}/{len(GROUP_LABELS)} "
                    f"coverage groups finished"
                )
            else:
                demo_progress_bar.progress(
                    fraction_done,
                    text=f"Generating test cases... {len(finished_groups)}/{len(GROUP_LABELS)} "
                         f"coverage groups complete ({elapsed}s elapsed)",
                )
                demo_caption.caption(
                    "Covering happy path, negative/boundary, business rules, "
                    "and failure/security scenarios in parallel."
                )

            if not drained_any:
                time.sleep(0.15)

        generation_thread.join()

        if not dev_mode:
            demo_progress_bar.progress(1.0, text="Generation complete.")

        if "error" in generation_outcome:
            st.error(f"Test case generation failed: {generation_outcome['error']}")
            st.stop()

        merged_result, group_results, duplicate_ids = generation_outcome["result"]

        generation_elapsed_time = round(time.time() - generation_start_time, 2)

        cache_hits = sum(1 for r in group_results if r["from_cache"])
        truncated_groups = [r for r in group_results if r["was_truncated"]]
        off_topic_dropped = sum(r["dropped_off_topic"] for r in group_results)
        copied_example_dropped = sum(r["dropped_copied_example"] for r in group_results)

        st.write(
            f"Test case generation completed in {generation_elapsed_time} seconds "
            f"across {len(group_results)} concurrent groups ({cache_hits} served from cache)."
        )

        if truncated_groups:
            st.warning(
                f"{len(truncated_groups)} group(s) appear to have been cut off mid-generation. "
                f"Affected groups: {', '.join(r['group_id'] for r in truncated_groups)}."
            )

        if off_topic_dropped:
            st.info(f"Dropped {off_topic_dropped} test case(s) generated for the wrong coverage area before merging.")

        if copied_example_dropped:
            st.warning(
                f"Dropped {copied_example_dropped} test case(s) that were near-identical "
                f"to the prompt's own worked example (the model recited the example "
                f"instead of using your actual requirement). If this keeps happening, "
                f"try a larger model or rerun - it's a known small-model failure mode."
            )

        if duplicate_ids:
            st.info(f"Removed {len(duplicate_ids)} near-duplicate test case(s): {', '.join(duplicate_ids)}.")

        failed_groups = [r for r in group_results if r["error"]]

        if failed_groups:
            with st.expander(f"{len(failed_groups)} group(s) needed repair / had issues"):
                for result in failed_groups:
                    st.write(f"**{result['group_id']}**: {result['error']}")
                    st.text(result["raw_text"][:1500])

        # Always write a full diagnostic log, regardless of pass/fail, so
        # a confusing result can be debugged from one attached file
        # instead of relying on screenshots of on-screen messages that
        # scroll out of view. This is the single most useful thing to
        # attach when reporting "fewer test cases than expected."
        debug_log_lines = [
            f"=== Generation Debug Log ===",
            f"Requirement ID: {active_requirement_id}",
            f"Generation model: {generation_model}",
            f"Total groups: {len(group_results)}",
            "",
        ]

        for result in group_results:
            debug_log_lines.append(f"--- Group: {result['group_id']} ---")
            debug_log_lines.append(f"from_cache: {result['from_cache']}")
            debug_log_lines.append(f"was_truncated: {result['was_truncated']}")
            debug_log_lines.append(f"dropped_off_topic: {result['dropped_off_topic']}")
            debug_log_lines.append(f"dropped_copied_example: {result['dropped_copied_example']}")
            debug_log_lines.append(f"error: {result['error']}")
            debug_log_lines.append(f"surviving_test_cases: {len(result['test_cases'])}")
            for tc in result["test_cases"]:
                debug_log_lines.append(f"  kept -> {tc.get('test_case_id', '')} | {tc.get('coverage_area', '')} | {tc.get('title', '')}")
            debug_log_lines.append("RAW MODEL OUTPUT (up to 4000 chars):")
            debug_log_lines.append(result["raw_text"][:4000])
            debug_log_lines.append("")

        debug_log_text = "\n".join(debug_log_lines)

        with open("outputs/generation_debug.log", "w", encoding="utf-8") as file:
            file.write(debug_log_text)

        with st.expander("Full generation debug log (download this if results look wrong)"):
            st.text(debug_log_text[:6000])

        st.download_button(
            label="Download Full Debug Log",
            data=debug_log_text,
            file_name="generation_debug.log",
            mime="text/plain",
            key="debug_log_download",
        )

        test_case_data = normalize_test_cases(merged_result, default_requirement_id=active_requirement_id)

        flagged_count = annotate_quality_flags(test_case_data["test_cases"])

        if flagged_count:
            with st.expander(
                f"{flagged_count} test case(s) flagged by deterministic quality checks "
                f"(contradictions, hedged outcomes, mismatched test type, thin steps)"
            ):
                st.caption(
                    "These are heuristic checks, not a guarantee of correctness - "
                    "review the flagged test cases yourself."
                )
                for test_case in test_case_data["test_cases"]:
                    if test_case.get("quality_flags"):
                        st.write(f"**{test_case.get('test_case_id', '')}** - {test_case.get('title', '')}")
                        for flag in test_case["quality_flags"]:
                            st.write(f"  - {flag}")

        try:

            validated_test_cases = TestCaseCollection(**test_case_data)
            final_json = validated_test_cases.model_dump_json(indent=4)

            with open("outputs/test_cases.json", "w", encoding="utf-8") as file:
                file.write(final_json)

            st.success(
                f"Generated {len(validated_test_cases.test_cases)} test cases "
                f"across {len(GROUP_LABELS)} coverage groups."
            )

            gap_report = build_gap_analysis_report(
                requirement, test_case_data["test_cases"], knowledge_folder="knowledge_base"
            )

            gap_report_markdown = format_report_as_markdown(gap_report, requirement_id=active_requirement_id)

            with open("outputs/coverage_gap_report.md", "w", encoding="utf-8") as file:
                file.write(gap_report_markdown)

            quality_report = ""

            if run_quality_score:

                st.subheader("Quality Scoring Report")
                quality_placeholder = st.empty()

                with st.spinner("Scoring generated test case quality..."):

                    prompt_template = open(
                        "prompts/quality_scoring_prompt.txt", "r", encoding="utf-8"
                    ).read()

                    computed_facts = format_facts_for_quality_prompt(gap_report)

                    quality_prompt = (
                        prompt_template
                        .replace("{requirement}", requirement[:1200])
                        .replace("{analysis}", analysis[:800])
                        .replace("{rag_context}", rag_context[:1000])
                        .replace("{test_cases}", final_json[:2500])
                        .replace("{computed_facts}", computed_facts)
                    )

                    quality_report = stream_text_response(quality_llm, quality_prompt, quality_placeholder)

                with open("outputs/quality_report.txt", "w", encoding="utf-8") as file:
                    file.write(quality_report)

            review = ""

            if run_reviewer and not fast_mode:

                st.subheader("Reviewer Feedback")
                review_placeholder = st.empty()

                with st.spinner("Reviewing test cases..."):

                    prompt_template = open(
                        "prompts/reviewer_prompt.txt", "r", encoding="utf-8"
                    ).read()

                    review_prompt = (
                        prompt_template
                        .replace("{requirement}", requirement[:1000])
                        .replace("{analysis}", analysis[:800])
                        .replace("{rag_context}", rag_context[:1000])
                        .replace("{test_cases}", final_json[:1500])
                    )

                    review = stream_text_response(review_llm, review_prompt, review_placeholder)

                with open("outputs/review.txt", "w", encoding="utf-8") as file:
                    file.write(review)

            export_start_time = time.time()

            output_files = save_outputs(validated_test_cases.test_cases, quality_report, "outputs")

            export_elapsed_time = round(time.time() - export_start_time, 2)
            st.write(f"Export completed in {export_elapsed_time} seconds.")

            with open("outputs/analysis.txt", "w", encoding="utf-8") as file:
                file.write(analysis)

            total_elapsed_time = round(time.time() - start_time, 2)

            st.success(f"Enterprise test cases generated successfully in {total_elapsed_time} seconds total.")

            if show_json:
                with st.expander("Generated JSON"):
                    st.json(test_case_data)

            with open(output_files["excel"], "rb") as file:
                excel_data = file.read()

            with open(output_files["word"], "rb") as file:
                word_data = file.read()

            # Persist everything to session_state so the Coverage & Gap and
            # Quality & Review tabs (and the top dashboard) can render it on
            # every subsequent rerun, not just inside this button's run.
            st.session_state["phase14_results"] = {
                "test_case_data": test_case_data,
                "gap_report": gap_report,
                "gap_report_markdown": gap_report_markdown,
                "quality_report": quality_report,
                "review": review,
                "excel_data": excel_data,
                "word_data": word_data,
                "active_requirement_id": active_requirement_id,
                "generation_elapsed_time": generation_elapsed_time,
                "total_elapsed_time": total_elapsed_time,
            }

            st.download_button(
                label="Download Excel", data=excel_data, file_name="test_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                label="Download Word Document", data=word_data, file_name="test_cases.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

            st.download_button(
                label="Download Coverage & Gap Analysis Report (Markdown)",
                data=gap_report_markdown, file_name="coverage_gap_report.md", mime="text/markdown",
            )

            st.info("See the **Coverage & Gap Analysis** and **Quality & Review** tabs for the full breakdown.")

        except json.JSONDecodeError as e:
            st.error("AI returned invalid JSON even after repair.")
            st.text(str(e))

        except ValidationError as e:
            st.error("JSON validation failed.")
            st.text(str(e))

    elif results:
        st.info(
            f"Showing results from the last run ({len(results['test_case_data']['test_cases'])} "
            f"test cases). Click the button above to generate again."
        )

    else:
        st.write("Click the button above to generate test cases once a requirement is available.")


# ---------------------------------------------------------------------------
# Tab 4 - Coverage & Gap Analysis (reads from session_state, persists across
# reruns/tab switches instead of disappearing once the button's run ends)
# ---------------------------------------------------------------------------

with tab_coverage:

    if not results:

        st.info("Generate test cases first - the coverage and gap analysis will appear here.")

    else:

        gap_report = results["gap_report"]

        st.caption(
            "Computed directly in Python from each test case's coverage_area, "
            "the requirement text, and the business rules knowledge base - "
            "not LLM self-reported."
        )

        summary_cols = st.columns(3)

        with summary_cols[0]:
            hit = len(ALL_COVERAGE_AREAS) - len(gap_report["missing_coverage_areas"])
            st.metric("Coverage Areas Hit", f"{hit}/{len(ALL_COVERAGE_AREAS)}")

        with summary_cols[1]:
            criteria_total = len(gap_report["criteria_traceability"])
            criteria_covered = criteria_total - len(gap_report["uncovered_criteria"])
            st.metric("Acceptance Criteria Covered", f"{criteria_covered}/{criteria_total}" if criteria_total else "N/A")

        with summary_cols[2]:
            rules_total = len(gap_report["business_rule_traceability"])
            rules_covered = rules_total - len(gap_report["uncovered_rules"])
            st.metric("Business Rules Covered", f"{rules_covered}/{rules_total}" if rules_total else "N/A")

        if gap_report["missing_coverage_areas"]:
            st.warning("Coverage areas with ZERO test cases: " + ", ".join(gap_report["missing_coverage_areas"]))

        st.subheader("Coverage Heatmap")

        st.plotly_chart(render_coverage_heatmap(gap_report["coverage_matrix"]), use_container_width=True)

        st.subheader("Acceptance Criteria Traceability")

        # status_label gives the shared pill renderer one consistent
        # field name to use across both criteria and rules tables.
        criteria_rows = [
            {**row, "status_label": "Covered" if row["covered"] else "MISSING"}
            for row in gap_report["criteria_traceability"]
        ]
        render_dataframe_with_pills(
            criteria_rows, text_col="text", status_col="status_label",
            ids_col="matched_test_case_ids", text_label="Acceptance Criterion",
        )

        st.subheader("Business Rule Traceability")

        rule_rows = [
            {**row, "status_label": "Covered" if row["covered"] else "MISSING"}
            for row in gap_report["business_rule_traceability"]
        ]
        render_dataframe_with_pills(
            rule_rows, text_col="text", status_col="status_label",
            ids_col="matched_test_case_ids", text_label="Business Rule",
        )

        st.subheader("Risk Coverage")

        risk_df = pd.DataFrame([
            {"Risk Tier": tier, "Test Case Count": len(ids), "Test Case IDs": ", ".join(ids) if ids else "-"}
            for tier, ids in gap_report["risk_coverage"].items()
        ])
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download Coverage & Gap Analysis Report (Markdown)",
            data=results["gap_report_markdown"], file_name="coverage_gap_report.md", mime="text/markdown",
            key="coverage_tab_download",
        )


# ---------------------------------------------------------------------------
# Tab 5 - Quality & Review
# ---------------------------------------------------------------------------

with tab_quality:

    if not results:

        st.info("Generate test cases first - the quality scoring and reviewer feedback will appear here.")

    else:

        if results["quality_report"]:
            st.subheader("Quality Scoring Report")
            st.markdown(results["quality_report"])
        else:
            st.caption("Quality Scoring Agent was not run for the last generation (check the sidebar option).")

        if results["review"]:
            st.subheader("Reviewer Feedback")
            st.markdown(results["review"])
        else:
            st.caption("Reviewer Agent was not run for the last generation (check the sidebar option).")
