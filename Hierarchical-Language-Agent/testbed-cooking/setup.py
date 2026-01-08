from setuptools import setup, find_packages

setup(name="gym_cooking",
      version="1.0.0",
      packages=find_packages(exclude=[]),
      python_requires='>=3.8',
      install_requires=[
          "gym",
          "numpy",
          "networkx",
          "matplotlib",
          "termcolor",
          "pygame",
      ])
