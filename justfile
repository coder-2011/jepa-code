set shell := ["zsh", "-lc"]

venv := ".venv"

test-batching:
    source {{venv}}/bin/activate && pytest tests/test_batching.py

test-benchmarking:
    source {{venv}}/bin/activate && pytest tests/test_benchmarking.py

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
    source {{venv}}/bin/activate && pytest tests/test_batching.py tests/test_benchmarking.py tests/test_embeddings.py tests/test_fineweb_dataloader.py tests/test_llm_jepa.py tests/test_masking.py tests/test_tokenization.py tests/test_train_checkpointing.py

download-fineweb-sample max_bytes="5242880" name="CC-MAIN-2024-10" output="tmp/fineweb-sample.jsonl":
    source {{venv}}/bin/activate && python scripts/download_fineweb_sample.py --name {{name}} --max-bytes {{max_bytes}} --output {{output}}

train-layer steps="50" batch_size="2" data_path="tmp/fineweb-sample.jsonl":
    source {{venv}}/bin/activate && python scripts/train.py --steps {{steps}} --batch-size {{batch_size}} --data-path {{data_path}}

train-llm-jepa steps="10" batch_size="1" max_length="256" model_name="hf-internal-testing/tiny-random-gpt2" train_file="llm-jepa/datasets/synth_train.jsonl" eval_file="llm-jepa/datasets/synth_test.jsonl":
    source {{venv}}/bin/activate && python scripts/train_llm_jepa.py --steps {{steps}} --batch-size {{batch_size}} --max-length {{max_length}} --model-name {{model_name}} --save-every 0 --train-file {{train_file}} --eval-file {{eval_file}}

train-llm-jepa-qwen steps="10" batch_size="1" max_length="256" train_file="llm-jepa/datasets/synth_train.jsonl" eval_file="llm-jepa/datasets/synth_test.jsonl" checkpoint_dir="checkpoints/llm-jepa-qwen":
    source {{venv}}/bin/activate && python scripts/train_llm_jepa.py --steps {{steps}} --batch-size {{batch_size}} --max-length {{max_length}} --model-name Qwen/Qwen3-0.6B --checkpoint-dir {{checkpoint_dir}} --save-every {{steps}} --train-file {{train_file}} --eval-file {{eval_file}} --wandb-mode offline

benchmark-openrouter dataset="llm-jepa/datasets/synth_test.jsonl" model="qwen/qwen3.5-397b-a17b" judge_model="qwen/qwen3.5-397b-a17b" max_examples="10" output="tmp/openrouter-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_openrouter.py --dataset {{dataset}} --model {{model}} --judge-model {{judge_model}} --max-examples {{max_examples}} --output {{output}} --force

benchmark-local-qwen dataset="llm-jepa/datasets/synth_test.jsonl" base_model="Qwen/Qwen3-0.6B" checkpoint="path/to/qwen-0.6b-checkpoint.pt" judge_model="openai/gpt-5.4" max_examples="10" device="mps" output="tmp/local-qwen-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model {{base_model}} --checkpoint {{checkpoint}} --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{output}} --force

benchmark-local-qwen-base dataset="llm-jepa/datasets/synth_test.jsonl" judge_model="openai/gpt-5.4" max_examples="10" device="mps" output="tmp/local-qwen-base-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint '' --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{output}} --force

benchmark-local-qwen-tuned dataset="llm-jepa/datasets/synth_test.jsonl" checkpoint="checkpoints/llm-jepa-qwen/latest.pt" judge_model="openai/gpt-5.4" max_examples="10" device="mps" output="tmp/local-qwen-tuned-benchmark.jsonl":
    source {{venv}}/bin/activate && python scripts/benchmark_local.py --dataset {{dataset}} --base-model Qwen/Qwen3-0.6B --checkpoint {{checkpoint}} --judge-model {{judge_model}} --max-examples {{max_examples}} --device {{device}} --output {{output}} --force
