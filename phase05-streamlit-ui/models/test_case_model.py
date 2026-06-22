# Import List so we can store multiple test steps
from typing import List

# Import BaseModel from Pydantic for JSON validation
from pydantic import BaseModel


# Model for one test case
class TestCase(BaseModel):

    # Unique test case ID
    test_case_id: str

    # Test case title
    title: str

    # Priority such as High, Medium, Low
    priority: str

    # Type such as Positive, Negative, Boundary
    test_type: str

    # Required condition before test execution
    preconditions: str

    # Test data used for execution
    test_data: str

    # List of test steps
    steps: List[str]

    # Expected result after execution
    expected_result: str

    # Requirement traceability ID
    requirement_id: str


# Model for multiple test cases
class TestCaseCollection(BaseModel):

    # List of TestCase objects
    test_cases: List[TestCase]