import asyncio
import json
import sys

from app.core.workflow import analyze, analyze_batch


def main():
    files = sys.argv[1:]
    if not files:
        print("usage: uv run main.py <file> [<file> ...]")
        return

    if len(files) == 1:
        result = asyncio.run(analyze(files[0]))
    else:
        result = asyncio.run(analyze_batch(files))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
