"""Checksum module."""

import multiprocessing
import os

import xxhash


def _file_checksum(file_path: str) -> str:
    """Compute checksum for a single file."""
    h = xxhash.xxh128()
    with open(file_path, "rb") as f:
        # Read in 64KB chunks
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"{h.hexdigest()}  {file_path}"


def calculate_checksum_(input_path: str, concurrency_number: int) -> str:
    """
    Compute checksums.

    Processes either a directory or an individual file, computes their
    checksums in parallel, and then computes a checksum of the checksums file.

    Args:
        input_path: Path to dir/file to process
        concurrency_number: Number of parallel jobs to use

    Returns:
        The checksum of all files combined
    """
    input_path = str(input_path)

    # Check if path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Error: '{input_path}' does not exist. Please check the path and try again."
        )

    # Collect files to process
    all_files = []
    if os.path.isfile(input_path):
        all_files = [input_path]
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                all_files.append(os.path.join(root, file))
        all_files.sort()
    else:
        raise FileNotFoundError(
            f"Error: '{input_path}' does not exist. Please check the path and try again."
        )

    # Run in parallel for multiple files
    with multiprocessing.Pool(processes=concurrency_number) as pool:
        checksums = pool.map(_file_checksum, all_files)

    h = xxhash.xxh128()
    for checksum in checksums:
        h.update((checksum + "\n").encode("utf-8"))
    return h.hexdigest()
