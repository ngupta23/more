import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="more",
    version="0.0.1b14",
    author="Nikhil Gupta",
    author_email="mywork.ng@gmail.com",
    description="A helper library for Pandas, Visualizations and Scikit-learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ngupta23/more",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)