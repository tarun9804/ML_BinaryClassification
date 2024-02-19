from setuptools import find_packages, setup


def get_packages(filepath):
    with open(filepath) as f:
        r = f.read()
        r_list = r.split("\n")


setup(name="Binary classification for admission data",
      version="0.0.1",
      description="predicting admission based on education params",
      author="Tarun",
      packages=find_packages(),
      install_requires=get_packages("requirements.txt")
      )
