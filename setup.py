import setuptools

with open("README.md","r") as f:
    long_description = f.read()

setuptools.setup(
    name="learnmolsim",
    version="0.1.0",
    author="Michael P. Howard",
    author_email="mphoward@auburn.edu",
    description="Learn molecular simulations, hands-on in Python!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mphowardlab/learnmolsim",
    project_urls={
        "Bug Tracker": "https://github.com/mphowardlab/learnmolsim/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
