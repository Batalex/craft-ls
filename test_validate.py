from jsonschema.validators import validator_for
import yaml
from jsonschema import ValidationError, validate
from jsonschema.validators import validator_for
import json


with open("schemas/snapcraft.json", "r") as f:
    schema = json.load(f)


with open("snapcraft.yaml", "r") as f:
    instance = yaml.safe_load(f)


validator_cls = validator_for(schema)
validator = validator_cls(schema)

errors = list(validator.iter_errors(instance))
print(errors)

# try:
#     validator = validator_for(schema)
#     validate(instance, schema)
# except ValidationError as excinfo:
#     print(excinfo)
#     print(excinfo.path)  # deque(['parts', 'my-part'])
#     print(excinfo.message)  # "111 is not of type 'object', 'null'"
