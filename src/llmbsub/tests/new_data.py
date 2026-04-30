#!/usr/bin/env python3
"""
Generate a fake dataset of JSONL files for testing.

Usage:
    python new_data.py <num_files> <total_size_mb> <path> [--depth N] [--workers N]

Examples:
    python new_data.py 10 100 /tmp/testdata           # 10 files totaling 100MB (flat)
    python new_data.py 100 1000 ./mydata              # 100 files totaling 1GB (flat)
    python new_data.py 50 500 /tmp/nested --depth 3   # 50 files in 3-level hierarchy
    python new_data.py 100 1000 /tmp/fast --workers 8 # Use 8 threads
"""

import argparse
import json
import os
import random
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def random_string(length: int) -> str:
    """Generate a random alphanumeric string."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_jsonl_record() -> dict:
    """Generate a random JSONL record with realistic-looking fields."""
    return {
        "id": random_string(12),
        "text": random_string(random.randint(100, 500)),
        "label": random.choice(["positive", "negative", "neutral"]),
        "score": round(random.uniform(0, 1), 4),
        "metadata": {
            "source": random_string(8),
            "timestamp": random.randint(1600000000, 1700000000),
            "tags": [random_string(6) for _ in range(random.randint(1, 5))],
        },
    }


def create_directory_structure(base_path: Path, depth: int) -> List[Path]:
    """
    Create nested directory structure and return list of all directories.

    Structure for depth=3:
        base/
        base/level_1/
        base/level_1/level_2/
        base/level_1/level_2/level_3/
    """
    dirs = [base_path]
    base_path.mkdir(parents=True, exist_ok=True)

    current = base_path
    for i in range(1, depth + 1):
        current = current / f"level_{i}"
        current.mkdir(exist_ok=True)
        dirs.append(current)

    return dirs


def generate_single_file(filename: Path, target_size: int) -> int:
    """Generate a single JSONL file. Returns bytes written."""
    file_written = 0
    with open(filename, "w") as f:
        while file_written < target_size:
            record = generate_jsonl_record()
            line = json.dumps(record) + "\n"
            f.write(line)
            file_written += len(line.encode("utf-8"))
    return file_written


def _worker_task(args: Tuple[str, int]) -> int:
    """Worker function for multiprocessing. Must be top-level for pickling."""
    filename_str, target_size = args
    return generate_single_file(Path(filename_str), target_size)


def generate_dataset(
    num_files: int,
    total_size_mb: float,
    output_path: str,
    depth: int = 0,
    workers: int = None,
):
    """Generate fake JSONL dataset files, optionally in nested directories."""
    output_dir = Path(output_path)

    # Default workers to min(num_files, cpu_count)
    if workers is None:
        workers = min(num_files, os.cpu_count() or 4)

    # Create directory structure
    if depth > 0:
        dirs = create_directory_structure(output_dir, depth)
        print(f"Created {len(dirs)} directories ({depth} levels deep)")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        dirs = [output_dir]

    total_bytes = int(total_size_mb * 1024 * 1024)
    bytes_per_file = total_bytes // num_files
    remainder = total_bytes % num_files

    print(f"Generating {num_files} JSONL files totaling {total_size_mb}MB in {output_path}")
    print(f"  Target bytes per file: ~{bytes_per_file:,}")
    print(f"  Using {workers} worker processes")
    if depth > 0:
        print(f"  Files distributed across {len(dirs)} directories")

    # Prepare file tasks: (filename_str, target_size) - use strings for pickling
    tasks: List[Tuple[str, int]] = []
    for i in range(num_files):
        target_dir = dirs[i % len(dirs)]
        filename = target_dir / f"data_{i:04d}.jsonl"
        target_size = bytes_per_file + (1 if i < remainder else 0)
        tasks.append((str(filename), target_size))

    start_time = time.time()
    total_written = 0
    completed = 0

    # Execute with process pool
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker_task, task) for task in tasks]
        for future in as_completed(futures):
            bytes_written = future.result()
            total_written += bytes_written
            completed += 1
            progress = completed / num_files * 100
            print(
                f"\r  Progress: {progress:5.1f}% ({completed}/{num_files} files)",
                end="",
                flush=True,
            )

    elapsed = time.time() - start_time
    throughput = total_written / 1024 / 1024 / elapsed if elapsed > 0 else 0

    print()
    print(
        f"Done. Total size: {total_written / 1024 / 1024:.2f}MB in {elapsed:.2f}s ({throughput:.1f} MB/s)"
    )

    # Show directory breakdown
    if depth > 0:
        print("\nFiles per directory:")
        for d in dirs:
            count = len(list(d.glob("*.jsonl")))
            rel_path = d.relative_to(output_dir.parent) if d != output_dir else d.name
            print(f"  {rel_path}: {count} files")


def main():
    parser = argparse.ArgumentParser(description="Generate fake JSONL dataset for testing")
    parser.add_argument("num_files", type=int, help="Number of JSONL files to create")
    parser.add_argument("total_size_mb", type=float, help="Total size in MB")
    parser.add_argument("path", help="Output directory path")
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=0,
        help="Directory nesting depth (0 = flat, 3 = 3 levels deep)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker threads (default: min(num_files, cpu_count))",
    )

    args = parser.parse_args()

    if args.num_files < 1:
        sys.exit("Error: num_files must be at least 1")
    if args.total_size_mb <= 0:
        sys.exit("Error: total_size_mb must be positive")
    if args.depth < 0:
        sys.exit("Error: depth must be non-negative")
    if args.workers is not None and args.workers < 1:
        sys.exit("Error: workers must be at least 1")

    generate_dataset(args.num_files, args.total_size_mb, args.path, args.depth, args.workers)


if __name__ == "__main__":
    main()
