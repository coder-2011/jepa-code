set shell := ["zsh", "-lc"]

venv := ".venv"

test-batching:
    source {{venv}}/bin/activate && pytest tests/test_batching.py

test-embeddings:
    source {{venv}}/bin/activate && pytest tests/test_embeddings.py

test-masking:
    source {{venv}}/bin/activate && pytest tests/test_masking.py

test-tokenization:
    source {{venv}}/bin/activate && pytest tests/test_tokenization.py

all-tests:
    source {{venv}}/bin/activate && pytest tests/test_batching.py tests/test_embeddings.py tests/test_masking.py tests/test_tokenization.py
