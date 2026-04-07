set shell := ["zsh", "-lc"]

venv := ".venv"

test-batching:
    source {{venv}}/bin/activate && pytest tests/test_batching.py

test-embeddings:
    source {{venv}}/bin/activate && pytest tests/test_embeddings.py

test-fineweb-dataloader:
    source {{venv}}/bin/activate && pytest tests/test_fineweb_dataloader.py

test-masking:
    source {{venv}}/bin/activate && pytest tests/test_masking.py

test-tokenization:
    source {{venv}}/bin/activate && pytest tests/test_tokenization.py

all-tests:
    source {{venv}}/bin/activate && pytest tests/test_batching.py tests/test_embeddings.py tests/test_fineweb_dataloader.py tests/test_masking.py tests/test_tokenization.py

download-fineweb-sample max_bytes="5242880" name="CC-MAIN-2024-10" output="tmp/fineweb-sample.jsonl":
    source {{venv}}/bin/activate && python scripts/download_fineweb_sample.py --name {{name}} --max-bytes {{max_bytes}} --output {{output}}
