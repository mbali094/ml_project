from zipfile import ZipFile
from pathlib import Path

def unzip_file(file_path:Path):

    with open(file_path, "r") as file:
        file.extractall("data")

