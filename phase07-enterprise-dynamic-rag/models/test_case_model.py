from typing import List
from pydantic import BaseModel


class TestCase(BaseModel):
    test_case_id: str
    title: str
    priority: str
    test_type: str
    preconditions: str
    test_data: str
    steps: List[str]
    expected_result: str
    requirement_id: str


class TestCaseCollection(BaseModel):
    test_cases: List[TestCase]