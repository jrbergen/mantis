"""Define schemas (column name -> data type mappings) for concrete datasets here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tmap_defectdetector.dataset.base.schemas_base import (
    SchemaLabels,
    SchemaEntry,
    SchemaSamplesImageData,
    SchemaFullImageData,
    SchemaSamples,
)


@dataclass(repr=False)
class SchemaLabelsELPV(SchemaLabels):
    """Specifies schema for ELPV label data."""

    LABEL_FILENAME: SchemaEntry = SchemaEntry(
        "LABEL_FILENAME",
        str,
        docstring="Name of the file which stores a row's data labels, as string.",
    )
    LABEL_PATH: SchemaEntry = SchemaEntry(
        "LABEL_PATH",
        str,
        docstring="Full path of the file which stores a row's data label, as string.",
    )
    TYPE: SchemaEntry = SchemaEntry(
        "TYPE",
        "category",
        docstring="Represents the type of photovoltaic panel (monocrystalline/polycrystalline).",
    )
    PROBABILITY: SchemaEntry = SchemaEntry(
        "PROBABILITY", np.float64, docstring="Expresses the degree of defectiveness."
    )


@dataclass(repr=False)
class SchemaSamplesELPV(SchemaSamplesImageData):
    """Specifies schema for ELPV sample data."""

    pass


@dataclass(repr=False)
class SchemaFullELPV(SchemaFullImageData, SchemaLabelsELPV, SchemaSamplesELPV):
    """Specifies schema for ELPV label _and_ sample data."""

    pass


@dataclass(repr=False)
class SchemaLabelsWineDetector(SchemaLabels):
    """Specifies schema for Wine Detector label data."""

    pass


@dataclass(repr=False)
class SchemaSamplesWineDetector(SchemaSamples):
    """Specifies schema for Wine Detector sample data."""

    pass


@dataclass(repr=False)
class SchemaFullWineDetector(SchemaSamples):
    """Specifies schema for Wine Detector label _and_ sample data"""

    pass
