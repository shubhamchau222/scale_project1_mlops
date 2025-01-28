import os
from pathlib import Path
import logging

# Doc link "https://packaging.python.org/en/latest/tutorials/packaging-projects/"

logging.basicConfig(
    level=logging.INFO,
    format= "[%(asctime)s: %(levelname)s]: %(message)s"
    )

while True:
    project_name = input("Enter the Project Name: ")
    if project_name != '':
        break

logging.info(f"Creating project by name: {project_name}")


# list of files:
list_of_files = [
    "./src/__init__.py",
    "./src/data_processing.py",
    "./src/model.py",
    "./data/raw/",
    "./models/",
    "./scripts/",
    "./tests/test_main.py",
    "__init__.py",
    "requirements.txt",
    "main.py",
    "train.py",
    "config.py",
    "buildspec.yml",
    "appspec.yml",
    "Dockerfile"


]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating a directory at: {filedir} for file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating a new file: {filename} at path: {filepath}")
    else:
        logging.info(f"file is already present at: {filepath}")



# tox.ini : help to check in various envs
# test: to test the packages 

# list of files:
# list_of_files = [
#     ".github/workflows/.gitkeep",
#     f"src/{project_name}/__init__.py",
#     f"tests/__init__.py",
#     f"tests/unit/__init__.py",
#     f"tests/integration/__init__.py",
#     "init_setup.sh",
#     "requirements.txt",
#     "requirements_dev.txt",
#     "setup.py",
#     "pyproject.toml",
#     'setup.cfg',
#     "tox.ini"
# ]



# tox.ini : help to check in various envs
# test: to test the packages 