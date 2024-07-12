#!/bin/bash

python3 -m venv tests/venv
source tests/venv/bin/activate
pip install uv
uv pip install msgpack-numpy pytest
