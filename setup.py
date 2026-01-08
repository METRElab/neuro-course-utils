from setuptools import setup, find_packages

setup(
    name="neuro_course_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "ipywidgets",
    ],
    author="Matin Yousefabaid",
    description="Interactive utilities for computational neuroscience course",
)
