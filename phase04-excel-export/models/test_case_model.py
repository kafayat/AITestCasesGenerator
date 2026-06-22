# Import List from typing
# List is used when one field can contain multiple values
# Example: steps = ["Step 1", "Step 2"]
from typing import List

# Import BaseModel from Pydantic
# Pydantic helps us create structured models and validate JSON data
from pydantic import BaseModel


# This class defines the structure of one test case
# Every generated test case must follow this format
class TestCase(BaseModel):

    # Unique ID for each test case
    # Example: TC001
    test_case_id: str

    # Short title of the test case
    # Example: Valid login with correct credentials
    title: str

    # Priority of the test case
    # Example: High, Medium, Low
    priority: str

    # Type/category of test case
    # Example: Positive, Negative, Boundary, Security
    test_type: str

    # Any condition required before executing the test
    # Example: User must already be registered
    preconditions: str

    # Test data required for execution
    # Example: valid username and password
    test_data: str

    # List of steps to execute the test case
    # Example: ["Open login page", "Enter username", "Click Login"]
    steps: List[str]

    # Expected system behaviour after executing the steps
    expected_result: str

    # Requirement ID used for traceability
    # Example: REQ001
    requirement_id: str


# This class represents a collection of multiple test cases
# The AI output will be validated against this structure
class TestCaseCollection(BaseModel):

    # List of test cases
    test_cases: List[TestCase]