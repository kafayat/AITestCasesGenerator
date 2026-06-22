# Import List so one field can contain multiple values
from typing import List

# Import Pydantic BaseModel for structured JSON validation
from pydantic import BaseModel


# This class represents one test case structure
class TestCase(BaseModel):

    # Unique test case ID
    test_case_id: str

    # Business-readable title
    title: str

    # Test priority such as High, Medium, Low
    priority: str

    # Test type such as Positive, Negative, Boundary, Security
    test_type: str

    # Conditions required before test execution
    preconditions: str

    # Test data used for test execution
    test_data: str

    # Test execution steps
    steps: List[str]

    # Expected result after executing the steps
    expected_result: str

    # Requirement ID for traceability
    requirement_id: str


# This class represents multiple test cases
class TestCaseCollection(BaseModel):

    # List of test case objects
    test_cases: List[TestCase]