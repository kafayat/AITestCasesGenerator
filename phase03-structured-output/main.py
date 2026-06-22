# Import JSON library
# Used for reading and writing JSON
import json

# Import Ollama LLM connector
from langchain_ollama import OllamaLLM

# Import validation error handling
from pydantic import ValidationError

# Import Requirement Analyzer Agent
from agents.requirement_analyzer_agent import analyze_requirement

# Import Test Case Generator Agent
from agents.test_case_generator_agent import generate_test_cases

# Import Reviewer Agent
from agents.reviewer_agent import review_test_cases

# Import Pydantic Models
from models.test_case_model import TestCaseCollection


# Create connection to local AI model
llm = OllamaLLM(
    model="llama3.2:3b"
)


# Open requirement file
with open(
    "requirement.txt",
    "r",
    encoding="utf-8"
) as file:

    # Read entire file content
    requirement = file.read()


# Run Requirement Analyzer Agent
print("Step 1 - Requirement Analysis Started")

analysis = analyze_requirement(
    llm,
    requirement
)

print("Step 1 - Requirement Analysis Completed")


# Run Test Case Generator Agent
print("Step 2 - Test Case Generation Started")

raw_test_cases = generate_test_cases(
    llm,
    requirement,
    analysis
)

print("Step 2 - Test Case Generation Completed")


# Validate generated JSON
try:

    # Convert string into JSON object
    test_case_data = json.loads(
        raw_test_cases
    )

    # Validate JSON structure
    validated_test_cases = TestCaseCollection(
        **test_case_data
    )

    # Convert object back to formatted JSON
    final_json = validated_test_cases.model_dump_json(
        indent=4
    )

    # Save final JSON file
    with open(
        "outputs/test_cases.json",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(final_json)

    print("JSON Validation Successful")

# Invalid JSON
except json.JSONDecodeError:

    print("AI returned invalid JSON")

    final_json = raw_test_cases


# JSON structure mismatch
except ValidationError as e:

    print("Pydantic Validation Failed")

    print(e)

    final_json = raw_test_cases


# Run Reviewer Agent
print("Step 3 - Review Started")

review = review_test_cases(
    llm,
    final_json
)

print("Step 3 - Review Completed")


# Save requirement analysis
with open(
    "outputs/analysis.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(analysis)


# Save reviewer feedback
with open(
    "outputs/review.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(review)


# Save raw AI output
with open(
    "outputs/raw_test_cases.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(raw_test_cases)


# Display outputs
print(final_json)

print(review)


# End of execution
print("Phase 3 Completed Successfully")