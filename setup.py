"""
We still need a setupl.py file (instead of only pyproject.toml for everything)
 until PEP660 gets implemented in setuptools if we want editable installs:
 https://github.com/pypa/setuptools/issues/2816.
 https://peps.python.org/pep-0660/
"""
from setuptools import setup, find_packages


def _read_readme(filename: str = "README.md") -> str:
    try:
        with open(filename, "r", encoding="utf-8") as readme_file_handle:
            return str(readme_file_handle.read())
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Readme file ({filename!r}) not found...") from err
    except PermissionError as err:
        raise PermissionError(f"No file permission to read readme file ({filename!r})...") from err
    except UnicodeEncodeError as err:
        raise UnicodeError(f"Unexpected encoding of readme file ({filename!r}); expected utf-8.") from err


def pkg_setup():
    setup(
        name="tmapdd",
        version="0.0.1",
        author="Multiple authors",
        author_email="author@example.com",
        description="Defect detection package using Tensorflow",
        long_description=_read_readme(),
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: UNIX",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.10",
        ],
        packages=find_packages("src"),
        package_dir={"": "src"},
        python_requires=">=3.10",
        entry_points={"console_scripts": ["defectdetector = tmap_defectdetector.main:main"]},
    )


if __name__ == "__main__":
    pkg_setup()
