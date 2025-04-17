from setuptools import setup, find_packages

setup(
    name="OPRDetectRecog",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
        "torch==2.,2.1",
        "torchvision==0.17.1",
        "deep_translator",
        "easyocr",
        "paddleocr",
        "paddlepaddle"
        "manga-ocr",
        "OperaPowerRelay @ git+https://github.com/OperavonderVollmer/OperaPowerRelay@main"
    ],
    python_requires=">=3.7",
    author="Opera von der Vollmer",
    description="Abstract Detector and Recognizer classes for consistency, used and made compatible for Opera's pipelines",
    url="https://github.com/OperavonderVollmer/OPRDetectRecog", 
    license="MIT",
)

