"""Manage server settings."""

import os

IS_DEV_MODE = os.environ.get("CRAFT_LS_DEV", False)
