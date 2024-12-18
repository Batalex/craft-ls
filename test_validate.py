from jsonschema.validators import validator_for
import yaml
from jsonschema import ValidationError, validate
from jsonschema.validators import validator_for
import json
from src.craft_ls.core import robust_load
from yaml.scanner import ScannerError
from yaml.events import (
    StreamEndEvent,
    MappingStartEvent,
    MappingEndEvent,
    DocumentEndEvent,
)


with open("schemas/snapcraft.json", "r") as f:
    schema = json.load(f)


# with open("snapcraft.yaml", "r") as f:
with open("tests/snapcraft.yaml", "r") as f:
    instance_document = f.read()


validator_cls = validator_for(schema)
validator = validator_cls(schema)


try:
    instance = robust_load(instance_document)
except:
    instance = yaml.safe_load(instance_document)
errors = list(validator.iter_errors(instance))
print(errors)


# try:
#     validator = validator_for(schema)
#     validate(instance, schema)
# except ValidationError as excinfo:
#     print(excinfo)
#     print(excinfo.path)  # deque(['parts', 'my-part'])
#     print(excinfo.message)  # "111 is not of type 'object', 'null'"
