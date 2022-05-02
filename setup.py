"""
We still need a setup.py file (instead of only pyproject.toml for everything)
until PEP660 gets implemented in setuptools if we want editable installs:
https://github.com/pypa/setuptools/issues/2816.
https://peps.python.org/pep-0660/
"""
import re
from pathlib import Path

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


def read_version_from_toml() -> str:
    """Hacky way to place version in 1 location with"""
    version_rex = re.compile(r"""version = \"(\d+\.\d+\.\d+)""")
    vpath: Path
    if not (vpath := Path(Path(__file__).parent, "pyproject.toml")).exists():
        raise FileNotFoundError("Couldn't find pyproject.toml file to read application version from...")
    try:
        return version_rex.search(vpath.read_text()).groups()[0]
    except (AttributeError, IndexError, KeyError) as err:
        raise NotImplementedError(
            "Version doesn't seem to be specified in pyproject.toml. "
            'Make sure to use the _exact_ syntax: version = "x.x.x" where x can be numbers)'
        ) from err


def pkg_setup():
    setup(
        name="mantis",
        version=read_version_from_toml(),
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
        entry_points={"console_scripts": [f"mantis = tmap_defectdetector.main:main"]},
    )


if __name__ == "__main__":
    pkg_setup()
