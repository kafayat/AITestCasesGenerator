# Import Pandas library
# Pandas helps us create tables and export Excel files
import pandas as pd


# Function to save generated test cases into Excel
def save_test_cases_to_excel(test_cases, output_file):

    # Empty list to store rows
    rows = []

    # Loop through all generated test cases
    for test_case in test_cases:

        # Convert TestCase object into dictionary format
        rows.append({

            # Excel Column: Test Case ID
            "Test Case ID": test_case.test_case_id,

            # Excel Column: Title
            "Title": test_case.title,

            # Excel Column: Priority
            "Priority": test_case.priority,

            # Excel Column: Test Type
            "Type": test_case.test_type,

            # Excel Column: Preconditions
            "Preconditions": test_case.preconditions,

            # Excel Column: Test Data
            "Test Data": test_case.test_data,

            # Convert step list into readable text
            # Each step appears on a new line
            "Steps": "\n".join(test_case.steps),

            # Expected Result column
            "Expected Result": test_case.expected_result,

            # Requirement Traceability
            "Requirement ID": test_case.requirement_id
        })

    # Convert list into Pandas DataFrame
    df = pd.DataFrame(rows)

    # Export DataFrame to Excel
    # index=False removes row numbering column
    df.to_excel(
        output_file,
        index=False
    )

    # Display success message
    print(f"Excel file saved successfully: {output_file}")