def load_prompt(prompt_file_path):

    with open(
        prompt_file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()