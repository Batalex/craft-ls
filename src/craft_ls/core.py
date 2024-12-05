"""Parser-validator core logic."""

import json
from importlib.resources import read_text

import yaml
from jsonschema import Validator
from jsonschema.validators import validator_for
from lsprotocol import types

validators: dict[str, Validator] = {}
for file_type in ["snapcraft"]:
    schema = json.loads(read_text("craft_ls.schemas", f"{file_type}.json"))
    validators[file_type] = validator_for(schema)(schema)


def get_diagnostics(
    validator: Validator, instance_document: str
) -> list[types.Diagnostic]:
    """Validate a document against its schema."""
    instance = yaml.safe_load(instance_document)
    diagnostics = []
    for error in validator.iter_errors(instance):
        diagnostics.append(
            types.Diagnostic(
                message=f"{error.message}",
                severity=types.DiagnosticSeverity.Information,
                # TODO(parse): get position from original yaml file
                range=types.Range(
                    start=types.Position(line=0, character=0),
                    end=types.Position(line=0, character=0),
                ),
            )
        )

    return diagnostics
