# Import json library
# Used to convert AI text output into JSON object
import json

# Import Ollama LLM connector from LangChain
# This allows Python to call local Ollama models
from langchain_ollama import OllamaLLM

# Import ValidationError from Pydantic
# Used to catch validation issues if JSON structure is wrong
from pydantic import ValidationError

# Import Requirement Analyzer Agent
from agents.requirement_analyzer_agent import analyze_requirement

# Import Test Case Generator Agent
from agents.test_case_generator_agent import generate_test_cases

# Import Reviewer Agent
from agents.reviewer_agent import review_test_cases

# Import Pydantic model for validating test case JSON
from models.test_case_model import TestCaseCollection

# Import Excel writer utility
from utils.excel_writer import save_test_cases_to_excel


# Create local LLM connection using Ollama
# llama3.2:3b is faster for local development
llm = OllamaLLM(
    model="llama3.2:3b"
)


# Read requirement from requirement.txt file
with open(
    "requirement.txt",
    "r",
    encoding="utf-8"
) as file:

    # Store requirement text in variable
    requirement = file.read()


# Step 1: Requirement Analysis
print("Step 1 - Requirement Analysis Started")

# Call Requirement Analyzer Agent
analysis = analyze_requirement(
    llm,
    requirement
)

print("Step 1 - Requirement Analysis Completed")


# Step 2: Test Case Generation
print("Step 2 - Test Case Generation Started")

# Call Test Case Generator Agent
raw_test_cases = generate_test_cases(
    llm,
    requirement,
    analysis
)

print("Step 2 - Test Case Generation Completed")


# Try to convert and validate AI output
try:

    # Convert raw AI string response into Python JSON/dictionary
    test_case_data = json.loads(
        raw_test_cases
    )

    # Validate JSON using Pydantic model
    # This ensures all required fields are present
    validated_test_cases = TestCaseCollection(
        **test_case_data
    )

    # Convert validated model back into formatted JSON text
    final_json = validated_test_cases.model_dump_json(
        indent=4
    )

    # Save validated JSON output
    with open(
        "outputs/test_cases.json",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(final_json)

    # Save validated test cases into Excel
    save_test_cases_to_excel(
        validated_test_cases.test_cases,
        "outputs/test_cases.xlsx"
    )

    print("JSON validation successful")
    print("Excel file created successfully")


# If AI output is not valid JSON
except json.JSONDecodeError:

    print("AI returned invalid JSON")

    # Print raw AI response for debugging
    print(raw_test_cases)

    # Store raw output so reviewer can still review it
    final_json = raw_test_cases


# If JSON is valid but does not match Pydantic model
except ValidationError as e:

    print("Pydantic validation failed")

    # Print validation error details
    print(e)

    # Store raw output so reviewer can still review it
    final_json = raw_test_cases


# Step 3: Review generated test cases
print("Step 3 - Reviewer Agent Started")

# Call Reviewer Agent
review = review_test_cases(
    llm,
    final_json
)

print("Step 3 - Reviewer Agent Completed")


# Save requirement analysis into text file
with open(
    "outputs/analysis.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(analysis)


# Save reviewer feedback into text file
with open(
    "outputs/review.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(review)


# Save raw AI output into text file
with open(
    "outputs/raw_test_cases.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(raw_test_cases)


# Final terminal messages
print("Phase 4 completed successfully")
print("Files created:")
print("outputs/test_cases.json")
print("outputs/test_cases.xlsx")
print("outputs/analysis.txt")
print("outputs/review.txt")
print("outputs/raw_test_cases.txt")