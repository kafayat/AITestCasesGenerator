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

    # Preconditions and test data can be plain text or structured JSON
    preconditions: Union[str, Dict[str, Any], List[Any]] = ""
    test_data: Union[str, Dict[str, Any], List[Any]] = ""

    # Test execution steps
    steps: List[str] = Field(default_factory=list)

    # Expected result can be plain text or structured output from the LLM
    expected_result: Union[str, Dict[str, Any], List[Any]] = ""

    # Automation suitability and source traceability
    automation_candidate: str = ""
    source_traceability: str = ""


# Container for all generated test cases
class TestCaseCollection(BaseModel):

    # List of generated test cases
    test_cases: List[TestCase] = Field(default_factory=list)