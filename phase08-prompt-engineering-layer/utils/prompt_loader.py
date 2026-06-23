# This utility loads prompt templates from the prompts folder
def load_prompt(prompt_file_path):

    # Open the prompt file
    with open(
        prompt_file_path,
        "r",
        encoding="utf-8"
    ) as file:

        # Read and return prompt text
        return file.read()