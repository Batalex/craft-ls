import yaml

with open("tests/snapcraft.yaml", "r") as f:
    tokens = list(yaml.scan(f))
