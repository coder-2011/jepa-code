set shell := ["zsh", "-lc"]

venv := ".venv"

test-batching:
    source {{venv}}/bin/activate && pytest tests/test_batching.py

test-embeddings:
    source {{venv}}/bin/activate && pytest tests/test_embeddings.py

test-fineweb-dataloader:
    source {{venv}}/bin/activate && pytest tests/test_fineweb_dataloader.py

test-llm-jepa:
    source {{venv}}/bin/activate && pytest tests/test_llm_jepa.py

test-masking:
    source {{venv}}/bin/activate && pytest tests/test_masking.py

test-tokenization:
    source {{venv}}/bin/activate && pytest tests/test_tokenization.py

test-train-checkpointing:
    source {{venv}}/bin/activate && pytest tests/test_train_checkpointing.py

all-tests:
    source {{venv}}/bin/activate && pytest tests/test_batching.py tests/test_embeddings.py tests/test_fineweb_dataloader.py tests/test_llm_jepa.py tests/test_masking.py tests/test_tokenization.py tests/test_train_checkpointing.py

download-fineweb-sample max_bytes="5242880" name="CC-MAIN-2024-10" output="tmp/fineweb-sample.jsonl":
    source {{venv}}/bin/activate && python scripts/download_fineweb_sample.py --name {{name}} --max-bytes {{max_bytes}} --output {{output}}

train-layer steps="50" batch_size="2" data_path="tmp/fineweb-sample.jsonl":
    source {{venv}}/bin/activate && python scripts/train.py --steps {{steps}} --batch-size {{batch_size}} --data-path {{data_path}}

train-llm-jepa steps="10" batch_size="1" max_length="256" train_file="llm-jepa/datasets/synth_train.jsonl" eval_file="llm-jepa/datasets/synth_test.jsonl":
    source {{venv}}/bin/activate && python scripts/train_llm_jepa.py --steps {{steps}} --batch-size {{batch_size}} --max-length {{max_length}} --save-every 0 --train-file {{train_file}} --eval-file {{eval_file}}
