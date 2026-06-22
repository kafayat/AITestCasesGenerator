from langchain_ollama import OllamaLLM

from agents.requirement_analyzer_agent import analyze_requirement
from agents.test_case_generator_agent import generate_test_cases
from agents.reviewer_agent import review_test_cases


# ---------------------------------------------------
# Load Local LLM Model
# ---------------------------------------------------

llm = OllamaLLM(model="llama3.2:3b")


# ---------------------------------------------------
# Read Requirement
# ---------------------------------------------------

with open("requirement.txt", "r", encoding="utf-8") as file:
    requirement = file.read()


# ---------------------------------------------------
# Step 1 - Requirement Analysis
# ---------------------------------------------------

print("Step 1 - Requirement Analysis Started...")

analysis = analyze_requirement(
    llm,
    requirement
)

print("Step 1 - Requirement Analysis Completed.")


# ---------------------------------------------------
# Step 2 - Test Case Generation
# ---------------------------------------------------

print("Step 2 - Test Case Generation Started...")

test_cases = generate_test_cases(
    llm,
    requirement,
    analysis
)

print("Step 2 - Test Case Generation Completed.")


# ---------------------------------------------------
# Step 3 - Review Test Cases
# ---------------------------------------------------

print("Step 3 - Review Started...")

review = review_test_cases(
    llm,
    test_cases
)

print("Step 3 - Review Completed.")


# ---------------------------------------------------
# Final Combined Output
# ---------------------------------------------------

final_output = f"""
==================================================
REQUIREMENT
==================================================

{requirement}

==================================================
REQUIREMENT ANALYSIS
==================================================

{analysis}

==================================================
GENERATED TEST CASES
==================================================

{test_cases}

==================================================
REVIEWER FEEDBACK
==================================================

{review}
"""


# ---------------------------------------------------
# Display Output
# ---------------------------------------------------

print(final_output)


# ---------------------------------------------------
# Save Outputs
# ---------------------------------------------------

with open("outputs/analysis.txt", "w", encoding="utf-8") as file:
    file.write(analysis)

with open("outputs/test_cases.txt", "w", encoding="utf-8") as file:
    file.write(test_cases)

with open("outputs/review.txt", "w", encoding="utf-8") as file:
    file.write(review)

with open(
    "outputs/phase2_multi_agent_output.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(final_output)


print("\nAll output files saved successfully.")
print("Location: outputs/")