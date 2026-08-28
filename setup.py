from setuptools import find_packages, setup

setup(
    name='mcqgenerator',
    version='0.0.1',
    author='parna jain',
    author_email='parnajain1412@gmail.com',
    install_requires=["langchain", "langchain-huggingface", "huggingface_hub", "streamlit", "python-dotenv", "PyPDF2"],
    packages=find_packages()
)