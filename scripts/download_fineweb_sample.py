from argparse import ArgumentParser
from pathlib import Path
import json

from datasets import load_dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--name", default="CC-MAIN-2024-10")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--output", default="tmp/fineweb-sample.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.name,
        split=args.split,
        streaming=True,
    )

    written_bytes = 0
    written_docs = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            text = (example.get("text") or "").strip()
            if not text:
                continue

            record = {
                "text": text,
                "id": example.get("id"),
                "dump": example.get("dump"),
                "url": example.get("url"),
                "date": example.get("date"),
                "language_score": example.get("language_score"),
                "token_count": example.get("token_count"),
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            encoded = line.encode("utf-8")
            handle.write(line)
            written_bytes += len(encoded)
            written_docs += 1

            if written_bytes >= args.max_bytes or written_docs >= args.max_docs:
                break

    print(f"wrote {written_docs} documents")
    print(f"wrote {written_bytes} bytes")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
