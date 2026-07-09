import io
import json

import pandas as pd

from docx import Document

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


def _extract_excel_structured(uploaded_file):
    """
    Phase 14 fix: the old handler called pd.read_excel(file).to_string()
    which only read the FIRST sheet and dumped raw cell values with NaN
    noise and no structure - giving the model almost nothing useful to
    generate from (confirmed by a real iBuy run that produced 4 generic
    test cases, 0 domain-specific content, 20% coverage score).

    This version:
    - Reads ALL sheets
    - Skips obviously irrelevant sheets (approval history, version logs)
    - For each sheet, detects whether it looks like a table (header row
      + data rows) or a key-value attribute list, and formats it
      accordingly as clean readable text
    - Strips NaN/empty cells so the model sees real content, not noise
    - Preserves sheet names as section headers so the model knows what
      each block of content is about
    """

    SKIP_SHEET_KEYWORDS = [
        "approval", "version", "history", "change log", "revision"
    ]

    try:
        file_bytes = uploaded_file.read()
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        uploaded_file.seek(0)
        xl = pd.ExcelFile(uploaded_file)

    sections = []

    for sheet_name in xl.sheet_names:

        sheet_lower = sheet_name.lower()

        if any(kw in sheet_lower for kw in SKIP_SHEET_KEYWORDS):
            continue

        try:
            df = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=sheet_name,
                header=None,
            )
        except Exception:
            continue

        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            continue

        rows = df.values.tolist()

        clean_rows = []
        for row in rows:
            clean = [
                str(cell).strip()
                for cell in row
                if str(cell).strip() not in ("", "nan", "NaN", "None")
            ]
            if clean:
                clean_rows.append(clean)

        if not clean_rows:
            continue

        section_lines = [f"--- {sheet_name.upper()} ---"]

        first_row = clean_rows[0]

        is_key_value = (
            len(first_row) <= 3
            and len(clean_rows) > 3
            and not any(
                kw in str(first_row).lower()
                for kw in ["order", "type", "mandatory", "field"]
            )
        )

        # Special handling for form-field sheets: scan first 5 rows for
        # a "Mandatory?" column header (handles merged-title Excel layouts
        # where row 0 is a sheet title and row 2 is the actual header).
        mandatory_col = None
        condition_col = None
        header_row_idx = None

        for row_idx in range(min(5, len(df))):
            raw_headers = [str(c).strip().lower() for c in df.iloc[row_idx].tolist()]
            if any("mandatory" in h for h in raw_headers):
                header_row_idx = row_idx
                mandatory_col = next(i for i, h in enumerate(raw_headers) if "mandatory" in h)
                condition_col = next(
                    (i for i, h in enumerate(raw_headers) if "instruction" in h or "condition" in h or "display" in h),
                    None
                )
                break

        if mandatory_col is not None and header_row_idx is not None:
            section_lines = [f"--- {sheet_name.upper()} ---"]
            try:
                df_form = pd.read_excel(
                    io.BytesIO(file_bytes),
                    sheet_name=sheet_name,
                    header=None,
                )
                for row_idx2, row in df_form.iterrows():
                    if row_idx2 <= header_row_idx:
                        continue  # skip title + header rows

                    ftype = str(row.iloc[1]).strip() if len(row) > 1 else ""
                    question = str(row.iloc[2]).strip() if len(row) > 2 else ""
                    mandatory = str(row.iloc[mandatory_col]).strip() if mandatory_col < len(row) else ""
                    condition = str(row.iloc[condition_col]).strip() if condition_col and condition_col < len(row) else ""

                    if ftype in ("nan", "NaN", "") or question in ("nan", "NaN", ""):
                        # Could be an option row (e.g. "Create", "Update", "Delete")
                        if question not in ("nan", "NaN", "") and ftype in ("nan", "NaN", ""):
                            section_lines.append(f"  Option: {question}")
                        continue

                    mandatory_text = "Mandatory: Yes" if mandatory == "Y" else "Mandatory: No" if mandatory in ("N", "") else ""
                    cond_text = f" | Display when: {condition}" if condition and condition not in ("nan", "NaN", "") else ""
                    section_lines.append(f"- [{ftype}] {question} | {mandatory_text}{cond_text}")

                sections.append("\n".join(section_lines))
                continue
            except Exception:
                pass  # fall through to generic handling

        if is_key_value:
            for row in clean_rows:
                if len(row) >= 2:
                    key = row[0]
                    value = row[1]
                    if key.lower() not in ("attribute", "name", "key"):
                        section_lines.append(f"{key}: {value}")
                elif len(row) == 1:
                    section_lines.append(row[0])
        else:
            for row in clean_rows:
                section_lines.append(" | ".join(row))

        sections.append("\n".join(section_lines))

    if not sections:
        try:
            df_fallback = pd.read_excel(io.BytesIO(file_bytes))
            return df_fallback.to_string()
        except Exception:
            return "Could not extract content from Excel file."

    # For Service Catalogue / form-spec Excel files, add inferred
    # acceptance criteria derived from the mandatory fields and actions
    # found in the form fields section - this gives the model explicit
    # testable statements to generate from, not just raw form field lists.
    full_text = "\n\n".join(sections)

    mandatory_fields = []
    actions = []
    conditional_fields = []

    for line in full_text.splitlines():
        if "Mandatory: Yes" in line:
            field = line.split("]")[1].split("|")[0].strip() if "]" in line else ""
            if field:
                mandatory_fields.append(field)
        if "Option: Create" in line or "Option: Update" in line or "Option: Delete" in line:
            action = line.replace("Option:", "").strip()
            if action not in actions:
                actions.append(action)
        if "Display when:" in line and "Mandatory: Yes" in line:
            conditional_fields.append(line.strip())

    if mandatory_fields or actions:
        criteria_lines = ["--- INFERRED ACCEPTANCE CRITERIA ---"]
        if actions:
            for action in actions:
                criteria_lines.append(f"- User must be able to submit a {action} request successfully.")
        if mandatory_fields:
            criteria_lines.append(f"- The following fields are mandatory and must be validated before submission: {', '.join(mandatory_fields[:10])}.")
        if conditional_fields:
            criteria_lines.append("- Conditional fields must only be shown when their display condition is met.")
            criteria_lines.append("- Conditional mandatory fields must be validated when shown.")
        criteria_lines.append("- System must not allow submission when mandatory fields are missing.")
        criteria_lines.append("- User must receive confirmation after successful submission.")
        criteria_lines.append("- System must route the request to the correct assignment group.")

        sections.append("\n".join(criteria_lines))

    return "\n\n".join(sections)


def extract_text_from_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    elif file_name.endswith(".pdf"):

        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    elif file_name.endswith(".docx"):

        document = Document(uploaded_file)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        return text

    elif file_name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)
        return df.to_string()

    elif file_name.endswith((".xlsx", ".xls", ".xlsm")):

        return _extract_excel_structured(uploaded_file)

    elif file_name.endswith(".json"):

        data = json.load(uploaded_file)
        return json.dumps(data, indent=2)

    else:
        return "Unsupported file type"
