#!/usr/bin/env python3
"""
Simple standalone test: Upload aborts when file is modified during upload.

Flow tested:
1. Calculate checksum using llmbsum.sh script
2. Start file upload (simulated with configurable delay)
3. Modify a file during the upload
4. Detect the modification and abort

Usage:
    python test_upload_abort_on_modify.py <target_directory> <upload_time_seconds>
    python test_upload_abort_on_modify.py --generate <num_files> <size_mb> <upload_time>

Examples:
    python test_upload_abort_on_modify.py /tmp/testdata 5.0
    python test_upload_abort_on_modify.py ./mydata 2.5
    python test_upload_abort_on_modify.py --generate 10 50 5.0           # Generate 10 files, 50MB, 5s upload
    python test_upload_abort_on_modify.py --generate 20 100 10.0 --depth 3  # Nested dirs
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

# =============================================================================
# CONFIGURATION - Modify these paths for your environment
# =============================================================================

# Path to checksum script - use env var if set, otherwise default to BlueVela path
LLMBSUM_SCRIPT_PATH = os.environ.get(
    "LLMBSUM_SCRIPT_PATH", "/proj/granite-build/tools/dirsum_sorted_128.sh"
)

# Concurrency for checksum calculation
CHECKSUM_CONCURRENCY = 8


# =============================================================================
# Enums and Data Classes
# =============================================================================


class UploadStatus(Enum):
    """Status of an upload operation."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class UploadResult:
    """Result of uploading a single input path."""

    path: str
    status: UploadStatus
    message: str
    checksum: Optional[str] = None


# =============================================================================
# Data Generation (calls new_data.py)
# =============================================================================

# Path to new_data.py script (same directory as this file)
NEW_DATA_SCRIPT = Path(__file__).parent / "new_data.py"


def generate_test_dataset(
    num_files: int, total_size_mb: float, output_path: str, depth: int = 0
):
    """Generate fake JSONL dataset files by calling new_data.py."""
    cmd = [
        sys.executable,
        str(NEW_DATA_SCRIPT),
        str(num_files),
        str(total_size_mb),
        output_path,
    ]
    if depth > 0:
        cmd.extend(["--depth", str(depth)])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"new_data.py failed with exit code {result.returncode}")

    return Path(output_path)


# =============================================================================
# Checksum Calculator
# =============================================================================


def get_directory_size(path: str) -> int:
    """Get total size of all files in directory (bytes)."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def calculate_checksum(path: str) -> Tuple[Optional[str], float]:
    """
    Calculate checksum for a directory using llmbsum.sh.

    Returns (checksum, elapsed_time) or (None, elapsed_time) on failure.
    """
    if not os.path.isfile(LLMBSUM_SCRIPT_PATH):
        print(f"  [Checksum] ERROR: llmbsum.sh not found at {LLMBSUM_SCRIPT_PATH}")
        return None, 0

    if not os.access(LLMBSUM_SCRIPT_PATH, os.X_OK):
        print(f"  [Checksum] ERROR: {LLMBSUM_SCRIPT_PATH} is not executable")
        return None, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "checksum")

        print(
            f"  [Checksum] Running: {LLMBSUM_SCRIPT_PATH} {path} {CHECKSUM_CONCURRENCY} {output_file}"
        )
        start_time = time.time()

        result = subprocess.run(
            [LLMBSUM_SCRIPT_PATH, path, str(CHECKSUM_CONCURRENCY), output_file],
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"  [Checksum] ERROR: Script failed (rc={result.returncode})")
            if result.stderr:
                print(f"  [Checksum] stderr: {result.stderr.strip()}")
            return None, elapsed

        # Check stderr for "command not found" errors (script may return 0 anyway)
        if result.stderr and "command not found" in result.stderr.lower():
            print(f"  [Checksum] ERROR: Missing dependency in llmbsum.sh")
            print(f"  [Checksum] stderr: {result.stderr.strip()}")
            return None, elapsed

        # Check that the intermediate .all file has content (parallel worked)
        all_file = f"{output_file}.all"
        if not os.path.exists(all_file):
            print(f"  [Checksum] ERROR: Intermediate file {all_file} not created")
            return None, elapsed

        all_file_size = os.path.getsize(all_file)
        if all_file_size == 0:
            print(
                f"  [Checksum] ERROR: Intermediate file is empty (parallel likely failed)"
            )
            if result.stderr:
                print(f"  [Checksum] stderr: {result.stderr.strip()}")
            return None, elapsed

        checksum_file = f"{output_file}.cksum"
        if not os.path.exists(checksum_file):
            print(f"  [Checksum] ERROR: Output file not created")
            return None, elapsed

        with open(checksum_file, "r") as f:
            checksum = f.read().strip().split()[0]

            # Benchmark output
            dir_size = get_directory_size(path)
            size_mb = dir_size / 1024 / 1024
            throughput = size_mb / elapsed if elapsed > 0 else 0
            file_count = sum(1 for _ in Path(path).rglob("*") if _.is_file())

            print(f"  [Benchmark] Checksum calculation:")
            print(f"    Concurrency: {CHECKSUM_CONCURRENCY} workers")
            print(f"    Data size:   {size_mb:.2f} MB ({file_count} files)")
            print(f"    Time:        {elapsed:.2f}s")
            print(f"    Throughput:  {throughput:.1f} MB/s")

            return checksum, elapsed


# =============================================================================
# Upload Service with Integrity Monitoring
# =============================================================================


class SimpleUploadService:
    """Upload service that monitors file integrity during upload."""

    def __init__(self, upload_delay: float, integrity_check_interval: float = 0.5):
        self.upload_delay = upload_delay
        self.integrity_check_interval = integrity_check_interval
        self._abort_requested = False

    def _get_all_mtimes(self, path: str) -> Dict[str, float]:
        """Get modification times for all files in path."""
        mtimes = {}
        p = Path(path)
        for f in p.rglob("*"):
            if f.is_file():
                mtimes[str(f)] = f.stat().st_mtime
        return mtimes

    def _monitor_integrity(
        self, path: str, initial_mtimes: Dict[str, float], stop_event: threading.Event
    ) -> bool:
        """Monitor source files for modifications during upload."""
        while not stop_event.is_set():
            current_mtimes = self._get_all_mtimes(path)

            if current_mtimes != initial_mtimes:
                changed = []
                for fp, mt in current_mtimes.items():
                    if fp not in initial_mtimes:
                        changed.append(f"  NEW: {fp}")
                    elif initial_mtimes[fp] != mt:
                        changed.append(f"  MODIFIED: {fp}")
                for fp in initial_mtimes:
                    if fp not in current_mtimes:
                        changed.append(f"  DELETED: {fp}")

                print(f"  [IntegrityMonitor] Source modified during upload!")
                for c in changed:
                    print(c)
                self._abort_requested = True
                return False

            stop_event.wait(self.integrity_check_interval)
        return True

    def process_input(self, path: str) -> UploadResult:
        """Process a single input path: checksum, upload with integrity monitoring."""
        print(f"\n[UploadService] Processing: {path}")
        self._abort_requested = False

        # Step 1: Calculate checksum
        print(f"  Step 1: Calculating checksum...")
        checksum, elapsed = calculate_checksum(path)

        if checksum is None:
            return UploadResult(
                path, UploadStatus.FAILED, "Checksum calculation failed"
            )

        print(f"  Checksum: {checksum} (calculated in {elapsed:.2f}s)")

        # Record initial mtimes
        initial_mtimes = self._get_all_mtimes(path)
        print(f"  Recorded mtimes for {len(initial_mtimes)} files")

        # Step 2: Start upload with integrity monitoring
        print(f"  Step 2: Starting upload ({self.upload_delay}s simulated)...")
        target_uri = f"lh://test/ns/filesets/input/{checksum}"

        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(
            target=self._monitor_integrity, args=(path, initial_mtimes, stop_monitor)
        )
        monitor_thread.start()

        # Simulate upload
        upload_done = threading.Event()
        upload_success = [False]

        def do_upload():
            print(f"  [Upload] Starting: {path} -> {target_uri}")
            time.sleep(self.upload_delay)
            upload_success[0] = True
            print(f"  [Upload] Complete")
            upload_done.set()

        upload_thread = threading.Thread(target=do_upload)
        upload_thread.start()

        # Wait for upload, checking abort flag
        while not upload_done.is_set():
            if self._abort_requested:
                print(f"  [UploadService] Abort detected!")
                break
            upload_done.wait(0.05)

        upload_thread.join(timeout=1.0)
        stop_monitor.set()
        monitor_thread.join(timeout=1.0)

        # Result
        if self._abort_requested:
            return UploadResult(
                path, UploadStatus.ABORTED, "Source modified during upload", checksum
            )
        elif upload_success[0]:
            return UploadResult(
                path, UploadStatus.SUCCESS, f"Uploaded to {target_uri}", checksum
            )
        else:
            return UploadResult(path, UploadStatus.FAILED, "Upload failed", checksum)


# =============================================================================
# Test Functions
# =============================================================================


def test_upload_with_modification(
    target_dir: str, upload_time: float, modify_after: float = None
):
    """Test upload with file modification during the process."""
    print("=" * 70)
    print("TEST: Upload with file modification detection")
    print("=" * 70)
    print(f"Target directory: {target_dir}")
    print(f"Upload time: {upload_time}s")

    if not os.path.isdir(target_dir):
        print(f"ERROR: Directory does not exist: {target_dir}")
        return False

    files = [f for f in Path(target_dir).rglob("*") if f.is_file()]
    print(f"Files: {len(files)}")

    if not files:
        print("ERROR: Directory is empty")
        return False

    target_file = files[0]
    if modify_after is None:
        modify_after = upload_time / 3

    print(f"Will modify after {modify_after}s: {target_file}")

    service = SimpleUploadService(
        upload_delay=upload_time, integrity_check_interval=0.2
    )

    def modify_file():
        time.sleep(modify_after)
        print(f"\n  >>> MODIFYING: {target_file.name} <<<")
        target_file.touch()

    modifier = threading.Thread(target=modify_file)
    modifier.start()

    result = service.process_input(target_dir)
    modifier.join()

    print("\n" + "=" * 70)
    print(f"RESULT: {result.status.value} - {result.message}")
    print("=" * 70)

    if result.status == UploadStatus.ABORTED:
        print("\n✓ PASS: Upload aborted on modification")
        return True
    else:
        print(f"\n✗ FAIL: Expected ABORTED, got {result.status.value}")
        return False


def test_upload_with_file_added(
    target_dir: str, upload_time: float, add_after: float = None
):
    """Test upload aborts when a new file is added during upload."""
    print("=" * 70)
    print("TEST: Upload with file addition detection")
    print("=" * 70)
    print(f"Target directory: {target_dir}")
    print(f"Upload time: {upload_time}s")

    if not os.path.isdir(target_dir):
        print(f"ERROR: Directory does not exist: {target_dir}")
        return False

    if add_after is None:
        add_after = upload_time / 3

    new_file = Path(target_dir) / "new_file_during_upload.txt"
    print(f"Will add file after {add_after}s: {new_file}")

    service = SimpleUploadService(
        upload_delay=upload_time, integrity_check_interval=0.2
    )

    def add_file():
        time.sleep(add_after)
        print(f"\n  >>> ADDING: {new_file.name} <<<")
        new_file.write_text("This file was added during upload")

    adder = threading.Thread(target=add_file)
    adder.start()

    result = service.process_input(target_dir)
    adder.join()

    # Clean up the added file
    if new_file.exists():
        new_file.unlink()

    print("\n" + "=" * 70)
    print(f"RESULT: {result.status.value} - {result.message}")
    print("=" * 70)

    if result.status == UploadStatus.ABORTED:
        print("\n✓ PASS: Upload aborted on file addition")
        return True
    else:
        print(f"\n✗ FAIL: Expected ABORTED, got {result.status.value}")
        return False


def test_upload_with_file_deleted(
    target_dir: str, upload_time: float, delete_after: float = None
):
    """Test upload aborts when a file is deleted during upload."""
    print("=" * 70)
    print("TEST: Upload with file deletion detection")
    print("=" * 70)
    print(f"Target directory: {target_dir}")
    print(f"Upload time: {upload_time}s")

    if not os.path.isdir(target_dir):
        print(f"ERROR: Directory does not exist: {target_dir}")
        return False

    files = [f for f in Path(target_dir).rglob("*") if f.is_file()]
    print(f"Files: {len(files)}")

    if not files:
        print("ERROR: Directory is empty")
        return False

    # Create a temporary file to delete (so we don't mess up the test data)
    temp_file = Path(target_dir) / "temp_file_to_delete.txt"
    temp_file.write_text("This file will be deleted during upload")

    if delete_after is None:
        delete_after = upload_time / 3

    print(f"Will delete file after {delete_after}s: {temp_file}")

    service = SimpleUploadService(
        upload_delay=upload_time, integrity_check_interval=0.2
    )

    def delete_file():
        time.sleep(delete_after)
        print(f"\n  >>> DELETING: {temp_file.name} <<<")
        temp_file.unlink()

    deleter = threading.Thread(target=delete_file)
    deleter.start()

    result = service.process_input(target_dir)
    deleter.join()

    # Clean up if file still exists (test failed)
    if temp_file.exists():
        temp_file.unlink()

    print("\n" + "=" * 70)
    print(f"RESULT: {result.status.value} - {result.message}")
    print("=" * 70)

    if result.status == UploadStatus.ABORTED:
        print("\n✓ PASS: Upload aborted on file deletion")
        return True
    else:
        print(f"\n✗ FAIL: Expected ABORTED, got {result.status.value}")
        return False


def test_upload_without_modification(target_dir: str, upload_time: float):
    """Test upload without modifications (should succeed)."""
    print("\n" + "=" * 70)
    print("TEST: Upload without modification")
    print("=" * 70)
    print(f"Target directory: {target_dir}")
    print(f"Upload time: {upload_time}s")

    if not os.path.isdir(target_dir):
        print(f"ERROR: Directory does not exist: {target_dir}")
        return False

    service = SimpleUploadService(
        upload_delay=upload_time, integrity_check_interval=0.2
    )
    result = service.process_input(target_dir)

    print("\n" + "=" * 70)
    print(f"RESULT: {result.status.value} - {result.message}")
    print("=" * 70)

    if result.status == UploadStatus.SUCCESS:
        print("\n✓ PASS: Upload succeeded")
        return True
    else:
        print(f"\n✗ FAIL: Expected SUCCESS, got {result.status.value}")
        return False


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test upload abort on file modification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with existing directory
  python test_upload_abort_on_modify.py /tmp/testdata 5.0

  # Generate test data in /tmp and run test
  python test_upload_abort_on_modify.py -g 10 50 5.0 -gd /tmp

  # Generate test data on GPFS
  python test_upload_abort_on_modify.py -g 10 1000 10 -gd /proj/granite-build/scratch

  # Generate nested test data
  python test_upload_abort_on_modify.py -g 20 100 10.0 -gd /tmp --depth 3

  # Test success case (no modification)
  python test_upload_abort_on_modify.py /tmp/testdata 2.0 --no-modify

  # Run all integrity tests (modify, add, delete, no-modify)
  python test_upload_abort_on_modify.py -g 10 50 5.0 -gd /tmp --all
        """,
    )

    # Mode 1: Use existing directory
    parser.add_argument(
        "target_directory",
        nargs="?",
        help="Directory to test (if not using --generate)",
    )
    parser.add_argument(
        "upload_time", nargs="?", type=float, help="Simulated upload time (seconds)"
    )

    # Mode 2: Generate test data
    parser.add_argument(
        "--generate",
        "-g",
        nargs=3,
        metavar=("NUM_FILES", "SIZE_MB", "UPLOAD_TIME"),
        help="Generate test data: num_files size_mb upload_time",
    )
    parser.add_argument(
        "--generate-dir",
        "-gd",
        metavar="DIR",
        help="Base directory for generated test data (required with --generate)",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=0,
        help="Directory nesting depth for generated data (0 = flat)",
    )

    # Test options
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all integrity tests (modify, add, delete, no-modify)",
    )
    parser.add_argument(
        "--no-modify", action="store_true", help="Test success case (no modification)"
    )
    parser.add_argument(
        "--modify-after",
        type=float,
        help="Seconds before modifying (default: upload_time/3)",
    )
    parser.add_argument(
        "--keep",
        "-k",
        action="store_true",
        help="Keep generated test data (don't clean up)",
    )

    args = parser.parse_args()

    print(f"\nChecksum script: {LLMBSUM_SCRIPT_PATH}")

    # Determine mode and parameters
    generated_dir = None
    cleanup_dir = None

    # Default: --generate 10 1000 10 --all -gd /tmp if no args provided
    # Also use defaults if only -gd is provided (use default generation params)
    use_defaults = (
        not args.generate and not args.target_directory and args.upload_time is None
    )
    if use_defaults:
        args.all = True  # Run all tests by default

    if args.generate or use_defaults or args.generate_dir:
        # Generate test data (use defaults if no -g args)
        if not args.generate:
            # No -g params, use defaults
            base_dir = args.generate_dir or "/tmp"
            print(f"[Default mode] Running: --generate 10 1000 10 --all -gd {base_dir}")
            num_files = 10
            size_mb = 1000.0
            upload_time = 10.0
            args.all = True  # Run all tests when using defaults
        else:
            num_files = int(args.generate[0])
            size_mb = float(args.generate[1])
            upload_time = float(args.generate[2])
            if not args.generate_dir:
                parser.error("--generate-dir (-gd) is required when using --generate")
            base_dir = args.generate_dir

        # Validate base directory exists
        if not os.path.isdir(base_dir):
            parser.error(f"Generate directory does not exist: {base_dir}")

        # Create temp directory for test data under the specified base
        generated_dir = tempfile.mkdtemp(prefix="upload_test_", dir=base_dir)
        if not args.keep:
            cleanup_dir = generated_dir

        print(f"\n[Setup] Generating test dataset in {generated_dir}")
        generate_test_dataset(num_files, size_mb, generated_dir, args.depth)

        target_directory = generated_dir

    else:
        # Mode 1: Use existing directory
        if not args.target_directory or args.upload_time is None:
            parser.error(
                "Either provide target_directory and upload_time, or use --generate"
            )

        target_directory = args.target_directory
        upload_time = args.upload_time

    try:
        # Run the tests
        if args.all:
            # Run all four integrity tests
            results = {}
            print("\n" + "=" * 70)
            print("RUNNING ALL INTEGRITY TESTS")
            print("=" * 70)

            print("\n[1/4] Testing file modification detection...")
            results["modify"] = test_upload_with_modification(
                target_directory, upload_time, args.modify_after
            )

            print("\n[2/4] Testing file addition detection...")
            results["add"] = test_upload_with_file_added(
                target_directory, upload_time, args.modify_after
            )

            print("\n[3/4] Testing file deletion detection...")
            results["delete"] = test_upload_with_file_deleted(
                target_directory, upload_time, args.modify_after
            )

            print("\n[4/4] Testing no modification (success case)...")
            results["no_modify"] = test_upload_without_modification(
                target_directory, upload_time
            )

            # Summary
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            for test_name, passed in results.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {test_name}: {status}")

            success = all(results.values())
            print(
                f"\nOverall: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}"
            )

        elif args.no_modify:
            success = test_upload_without_modification(target_directory, upload_time)
        else:
            success = test_upload_with_modification(
                target_directory, upload_time, args.modify_after
            )

    finally:
        # Cleanup generated data
        if cleanup_dir and os.path.exists(cleanup_dir):
            print(f"\n[Cleanup] Removing generated test data: {cleanup_dir}")
            shutil.rmtree(cleanup_dir)
        elif generated_dir and args.keep:
            print(f"\n[Keep] Test data preserved at: {generated_dir}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
