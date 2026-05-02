"""
reduce the logs into three numbers: average latency_ps, average fidelity_corrected, and number of rows processed
"""

import argparse
import re
from pathlib import Path


LINE_RE = re.compile(r"\blatency_ps=(?P<latency>\S+)\s+fidelity_corrected=(?P<fidelity>\S+)")


def resolve_input_dir(dir_arg: str) -> Path:
    for candidate in (Path(dir_arg), Path("log") / dir_arg):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Directory not found: {dir_arg}")


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_processed")


def process_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    processed = 0
    skipped = 0
    latency_sum = 0.0
    fidelity_sum = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="replace") as src:
        with output_path.open("w", encoding="utf-8") as dst:
            for line in src:
                match = LINE_RE.search(line)
                if not match:
                    if line.strip():
                        skipped += 1
                    continue

                latency_sum += float(match.group("latency"))
                fidelity_sum += float(match.group("fidelity"))
                processed += 1

            if processed:
                dst.write(f"{latency_sum / processed}, {fidelity_sum / processed}, {processed}\n")

    return processed, skipped


def process_dir(input_dir: Path, output_dir: Path) -> tuple[int, int, int]:
    files = rows = skipped_lines = 0

    for input_path in input_dir.rglob("*"):
        if not input_path.is_file():
            continue

        processed, skipped = process_file(input_path, output_dir / input_path.relative_to(input_dir))
        files += 1
        rows += processed
        skipped_lines += skipped

    return files, rows, skipped_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact QEC logs to 'latency_ps, fidelity_corrected' rows.")
    parser.add_argument("--dir", help="Input directory path or name under ./log.")
    parser.add_argument("--output-dir", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = resolve_input_dir(args.dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_dir)

    files, rows, skipped = process_dir(input_dir, output_dir)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files processed: {files}")
    print(f"Rows processed: {rows}")
    if skipped:
        print(f"Non-matching non-empty lines skipped: {skipped}")


if __name__ == "__main__":
    main()
