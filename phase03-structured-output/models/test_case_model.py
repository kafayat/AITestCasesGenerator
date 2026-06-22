# Import List datatype
# Used when a field contains multiple items
from typing import List

# Import BaseModel from Pydantic
# BaseModel allows us to create structured data models
from pydantic import BaseModel


# Define a single Test Case model
# This acts like a template/schema
class TestCase(BaseModel):

    # Unique Test Case ID
    test_case_id: str

    # Name of the test case
    title: str

    # Priority of execution
    # Example: High, Medium, Low
    priority: str

    # Type of test case
    # Example: Positive, Negative, Boundary
    test_type: str

    # Conditions required before running test
    preconditions: str

    # Test data used in execution
    test_data: str

    # List of test execution steps
    steps: List[str]

    # Expected outcome after execution
    expected_result: str

    # Requirement traceability ID
    requirement_id: str


# Collection model
# Used when AI generates multiple test cases
class TestCaseCollection(BaseModel):

    # Store all generated test cases
    test_cases: List[TestCase]