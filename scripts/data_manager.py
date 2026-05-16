import os
import json
from typing import Any
from utils.helpers import read_json, write_json

def load_products(path: str):
    return read_json(path)

def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json(path, data)

def path_exists(path: str) -> bool:
    return os.path.exists(path)
