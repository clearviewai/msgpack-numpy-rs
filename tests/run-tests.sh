#!/bin/bash

echo "Performing unit tests..."
cargo test --lib

echo "Testing deserializing in Rust msgpacks created in Python..."
cargo test --test test_deserialize

echo "Testing deserializing in Python msgpacks created in Rust..."
source tests/venv/bin/activate
pytest tests/test_deserialize.py -s
