import os


def save_uploaded_file(uploaded_file, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    file_path = os.path.join(
        target_folder,
        uploaded_file.name
    )

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def read_txt_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")