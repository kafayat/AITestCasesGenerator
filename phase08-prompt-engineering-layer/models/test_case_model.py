from typing import List
from pydantic import BaseModel


class TestCase(BaseModel):
    test_case_id: str
    requirement_id: str = ""
    requirement_reference: str = ""
    business_rule_reference: str = ""
    module: str = ""
    feature: str = ""
    risk_level: str = ""
    test_objective: str = ""
    title: str
    priority: str
    test_type: str
    preconditions: str
    test_data: str
    steps: List[str]
    expected_result: str
    post_condition: str = ""
    automation_candidate: str = ""
    traceability: str = ""


class TestCaseCollection(BaseModel):
    test_cases: List[TestCase]