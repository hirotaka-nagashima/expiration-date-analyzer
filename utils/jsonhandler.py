"""Handles json files."""

import json
import os

ENCODING = "utf-8-sig"  # for Japanese


def load(src):
    if not os.path.exists(src):
        return {}
    with open(src, "r", encoding=ENCODING) as file:
        data = json.load(file)
    return data


def dump(data, dest):
    with open(dest, "w", encoding=ENCODING) as file:
        def to_str(o):
            return getattr(o, "isoformat", getattr(o, "__str__", None))()
        json.dump(data, file, ensure_ascii=False, indent=2, default=to_str)


def update(data, dest):
    existing_data = load(dest)
    existing_data.update(data)
    dump(existing_data, dest)
