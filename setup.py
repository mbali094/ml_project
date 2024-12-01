from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:

    '''
    This function will take in a requirement file and return a list of libraries.
    '''

    with open(file_path, "r") as f:
        requirements=f.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements
        

setup(

    name="ml_project",
    version="0.0.1",
    author = "mabli094",
    author_email = "mbalinene094@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)