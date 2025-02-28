"""Types module."""

from dataclasses import dataclass

from yaml import Token


@dataclass
class ScanResult:
    """Token scan result container."""

    tokens: list[Token]


@dataclass
class CompleteScan(ScanResult):
    """Indicate a complete scan of the document."""

    pass


@dataclass
class IncompleteScan(ScanResult):
    """Indicate an incomplete scan of the document."""

    pass
