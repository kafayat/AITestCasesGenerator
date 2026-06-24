from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    test_case_id: str = ""
    requirement_id: str = ""
    title: str = ""
    priority: str = ""
    test_type: str = ""

    preconditions: Union[str, Dict[str, Any]] = ""
    test_data: Union[str, Dict[str, Any]] = ""

    steps: List[str] = Field(default_factory=list)

    expected_result: str = ""
    automation_candidate: str = ""
    source_traceability: str = ""


class TestCaseCollection(BaseModel):
    test_cases: List[TestCase] = Field(default_factory=list)