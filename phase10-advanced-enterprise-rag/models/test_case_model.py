from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field


# Represents a single generated test case
class TestCase(BaseModel):

    # Basic test case information
    test_case_id: str = ""
    requirement_id: str = ""
    title: str = ""
    priority: str = ""
    test_type: str = ""

    # Preconditions and test data
    # Can be text or structured JSON
    preconditions: Union[str, Dict[str, Any]] = ""
    test_data: Union[str, Dict[str, Any]] = ""

    # Test execution steps
    steps: List[str] = Field(default_factory=list)

    # Expected outcome and traceability
    expected_result: str = ""
    automation_candidate: str = ""
    source_traceability: str = ""


# Container for all generated test cases
class TestCaseCollection(BaseModel):

    # List of generated test cases
    test_cases: List[TestCase] = Field(default_factory=list)