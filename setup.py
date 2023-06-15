from setuptools import find_packages, setup
from typing import List

hypen_e_dot = '-e .'


#below function returns a list of packages which needs to be installed
def get_requirements(filepath:str):
    required_modules = []
    with open(filepath) as file_opener:
        required_modules = file_opener.readlines()
        required_modules = [modules.replace('\n', '') for modules in required_modules]

    if hypen_e_dot in required_modules:
        required_modules.remove(hypen_e_dot)
    
    return required_modules

#below this code set basic information for project creation like name, version, author name, email etc.
setup(
    name='score-predictor', 
    version='0.0.0',
    author='Deepak',
    author_email='deepaksainiofficial17@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)