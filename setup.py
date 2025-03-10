from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medtranscluster",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Medical Image Transfer Learning and Clustering Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MedTransCluster",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "tensorflow>=2.4.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=4.14.0",
        "pillow>=8.0.0",
        "tqdm>=4.50.0",
    ],
) 