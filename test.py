import yaml
from jsonschema import validate
import json


with open("schemas/snapcraft.json", "r") as f:
    schema = json.load(f)


with open("snapcraft.yaml", "r") as f:
    instance = yaml.safe_load(f)

validate(instance, schema)
