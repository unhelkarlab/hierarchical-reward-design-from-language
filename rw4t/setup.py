from setuptools import setup, find_packages

# with open("README.md", "r", encoding="utf-8") as fh:
#   long_description = fh.read()

setup(name="rw4t",
      version="0.0.1",
      long_description_content_type="text/markdown",
      packages=find_packages(exclude=[]),
      python_requires='>=3.8',
      install_requires=[])
