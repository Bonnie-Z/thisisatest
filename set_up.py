from setuptools import setup, find_packages

setup(
    name = "data_cleaning",
    version = "0.1",
    packages = find_packages(),
    package_data = {"data_cleaning.data": ['*.csv'],},
    install_requires = ["pandas", "numpy",],
    entry_points = {"console_scripts": ["clean-data = data_cleaning.cleaning:main", ],},
)







