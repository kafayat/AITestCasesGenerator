# Import pandas for creating Excel file
import pandas as pd


# Function to save test cases into Excel
def save_test_cases_to_excel(test_cases, output_file):

    # Create empty rows list
    rows = []

    # Loop through each test case
    for test_case in test_cases:

        # Add each test case as one Excel row
        rows.append({
            "Test Case ID": test_case.test_case_id,
            "Title": test_case.title,
            "Priority": test_case.priority,
            "Type": test_case.test_type,
            "Preconditions": test_case.preconditions,
            "Test Data": test_case.test_data,
            "Steps": "\n".join(test_case.steps),
            "Expected Result": test_case.expected_result,
            "Requirement ID": test_case.requirement_id
        })

    # Convert list into dataframe
    df = pd.DataFrame(rows)

    # Save dataframe to Excel
    df.to_excel(output_file, index=False)