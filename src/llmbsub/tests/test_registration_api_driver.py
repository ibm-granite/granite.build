#!/usr/bin/env python3
"""
Test driver for artifact upload workflow with mock registration API.

Tests two scenarios:
1. New artifact: Checksum not found in registry -> upload via DMF CLI
2. Existing artifact: Checksum found -> skip upload

This driver uses a mock API to simulate gbserver's /artifact/status endpoint,
allowing testing of the upload flow without a live server.

Usage:
    # Test new artifact case (checksum not in mock registry)
    python test_registration_api_driver.py /path/to/dataset --space my_space

    # Test existing artifact case (pre-register checksum)
    python test_registration_api_driver.py /path/to/dataset --space my_space --simulate-existing

    # With actual DMF upload
    python test_registration_api_driver.py /path/to/dataset --space my_space --upload

Examples:
    # Run on BlueVela with test data generation
    python test_registration_api_driver.py -gd /proj/granite-build/scratch

    # Use existing dataset
    python test_registration_api_driver.py /proj/granite-build/scratch/mydata --space granite_space
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to checksum script - use env var if set, otherwise default to BlueVela path
CHECKSUM_SCRIPT_PATH = os.environ.get(
    "LLMBSUM_SCRIPT_PATH", "/proj/granite-build/tools/dirsum_sorted_128.sh"
)

# Concurrency for checksum calculation
CHECKSUM_CONCURRENCY = 8

# Default DMF namespace
DEFAULT_NAMESPACE = "granite_dot_build.public"

# Path to new_data.py script (same directory as this file)
NEW_DATA_SCRIPT = Path(__file__).parent / "new_data.py"


# =============================================================================
# Mock Registration API
# =============================================================================


@dataclass
class ArtifactInfo:
    """Information about a registered artifact."""

    checksum: str
    space_name: str
    uri: str
    uuid: str
    origin_uri: str


class MockRegistrationAPI:
    """Mock artifact registration API for testing.

    Simulates the gbserver /artifact/status and /artifact/lh/fileset endpoints.
    Based on PR #1660 which adds checksum field to artifacts.
    """

    def __init__(self):
        self.registered_artifacts: Dict[str, ArtifactInfo] = (
            {}
        )  # checksum -> artifact_info
        self.call_history = []

    def check_artifact(self, checksum: str, space_name: str) -> dict:
        """Check if artifact with checksum exists.

        Returns:
            {"status": "exists", "uri": "lh://...", "uuid": "..."} if found
            {"status": "not_found", "target_uri": "lh://..."} if not found
        """
        self.call_history.append(
            {"method": "check_artifact", "checksum": checksum, "space_name": space_name}
        )

        if checksum in self.registered_artifacts:
            artifact = self.registered_artifacts[checksum]
            return {"status": "exists", "uri": artifact.uri, "uuid": artifact.uuid}
        else:
            # Generate target URI for new upload
            target_uri = f"lh://prod/{space_name}/filesets/input/{checksum}"
            return {"status": "not_found", "target_uri": target_uri}

    def register_artifact(
        self,
        checksum: str,
        space_name: str,
        origin_uri: str,
        namespace: str,
        label: str,
    ) -> dict:
        """Register new artifact.

        Returns:
            {"status": "registered", "uri": "lh://...", "uuid": "..."} on success
            {"status": "conflict", "existing_uri": "...", "existing_uuid": "..."} if checksum exists
        """
        self.call_history.append(
            {
                "method": "register_artifact",
                "checksum": checksum,
                "space_name": space_name,
                "origin_uri": origin_uri,
            }
        )

        if checksum in self.registered_artifacts:
            existing = self.registered_artifacts[checksum]
            return {
                "status": "conflict",
                "existing_uri": existing.uri,
                "existing_uuid": existing.uuid,
            }

        # Register new artifact
        artifact_uuid = str(uuid.uuid4())
        artifact_uri = f"lh://prod/{namespace}/filesets/{label}/{checksum}"

        artifact = ArtifactInfo(
            checksum=checksum,
            space_name=space_name,
            uri=artifact_uri,
            uuid=artifact_uuid,
            origin_uri=origin_uri,
        )
        self.registered_artifacts[checksum] = artifact

        return {"status": "registered", "uri": artifact_uri, "uuid": artifact_uuid}

    def pre_register(self, checksum: str, space_name: str, uri: str = None):
        """Pre-register an artifact for testing existing artifact case."""
        artifact_uuid = str(uuid.uuid4())
        if uri is None:
            uri = f"lh://prod/{space_name}/filesets/input/{checksum}"

        artifact = ArtifactInfo(
            checksum=checksum,
            space_name=space_name,
            uri=uri,
            uuid=artifact_uuid,
            origin_uri=f"file:///original/path",
        )
        self.registered_artifacts[checksum] = artifact


# =============================================================================
# Data Generation
# =============================================================================


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
# Checksum Calculation
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
    Calculate checksum for a directory using dirsum_sorted.sh.

    Returns (checksum, elapsed_time) or (None, elapsed_time) on failure.
    """
    if not os.path.isfile(CHECKSUM_SCRIPT_PATH):
        print(f"  [Checksum] ERROR: Script not found at {CHECKSUM_SCRIPT_PATH}")
        return None, 0

    if not os.access(CHECKSUM_SCRIPT_PATH, os.X_OK):
        print(f"  [Checksum] ERROR: {CHECKSUM_SCRIPT_PATH} is not executable")
        return None, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "checksum")

        print(
            f"  [Checksum] Running: {CHECKSUM_SCRIPT_PATH} {path} {CHECKSUM_CONCURRENCY} {output_file}"
        )
        start_time = time.time()

        result = subprocess.run(
            [CHECKSUM_SCRIPT_PATH, path, str(CHECKSUM_CONCURRENCY), output_file],
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

        # Check stderr for "command not found" errors
        if result.stderr and "command not found" in result.stderr.lower():
            print(f"  [Checksum] ERROR: Missing dependency")
            print(f"  [Checksum] stderr: {result.stderr.strip()}")
            return None, elapsed

        # Check that the intermediate .all file has content
        all_file = f"{output_file}.all"
        if not os.path.exists(all_file):
            print(f"  [Checksum] ERROR: Intermediate file {all_file} not created")
            return None, elapsed

        all_file_size = os.path.getsize(all_file)
        if all_file_size == 0:
            print(f"  [Checksum] ERROR: Intermediate file is empty")
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
# Build YAML Generation
# =============================================================================


def generate_build_yaml(
    output_dir: str,
    input_name: str,
    origin_path: str,
    checksum: str,
    lakehouse_uri: str,
    job_id: str = "test_job_001",
    build_name: str = None,
) -> Path:
    """
    Generate a build.yaml file with input artifact metadata.

    Uses the new format where:
    - Primary URI is env:// (original path)
    - metadata.checksum contains the calculated checksum
    - metadata.other_locations contains the Lakehouse URI(s)

    The env:// path is also stored in origin_uris in ArtifactRegistration.

    Args:
        output_dir: Directory to write build.yaml
        input_name: Name of the input artifact
        origin_path: Original path to the dataset (will be converted to env://)
        checksum: Calculated checksum of the dataset
        lakehouse_uri: Lakehouse URI from artifact registration
        job_id: BSub job ID (for build naming)
        build_name: Optional build name override

    Returns:
        Path to the generated build.yaml file
    """
    # Convert path to env:// URI
    env_uri = f"env://{os.path.abspath(origin_path)}"

    build_inputs = {
        input_name: {
            "uri": env_uri,
            "metadata": {
                "checksum": checksum,
                "other_locations": [{"uri": lakehouse_uri}],
            },
        }
    }

    build_outputs = {}  # No outputs for this test

    step = {
        "step_uri": "space://steps/env_exec",
        "config": {
            "lsf": {
                "bsub": {
                    "jobid": job_id,
                    "log_path": f"/tmp/logs/{job_id}.log",
                }
            },
        },
    }

    target = {
        "environment_uri": "space://environments/bluevela",
        "inputs": build_inputs,
        "outputs": build_outputs,
        "steps": [step],
    }

    build_config = {
        "granite.build": {
            "name": build_name or f"bluevela_job_{job_id}",
            "targets": {"my_workload": target},
        }
    }

    output_path = Path(output_dir)
    build_yaml_path = output_path / f"build-{job_id}.yaml"

    with open(build_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            build_config,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    return build_yaml_path


# =============================================================================
# DMF Upload
# =============================================================================


def run_dmf_upload(
    directory: str, namespace: str, label: str, version: str, dry_run: bool = True
) -> bool:
    """
    Run DMF fileset push command.

    Syntax: dmf fileset push <label> -n <namespace> -d <directory> --version <version>

    Returns True on success, False on failure.
    """
    cmd = [
        "dmf",
        "fileset",
        "push",
        label,
        "-n",
        namespace,
        "-d",
        directory,
        "--version",
        version,
    ]

    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    print(f"  [Upload] Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for large uploads
        )

        if result.returncode == 0:
            print(f"  [Upload] Success!")
            if result.stdout:
                print(f"  [Upload] stdout: {result.stdout.strip()}")
            return True
        else:
            print(f"  [Upload] FAILED (rc={result.returncode})")
            if result.stderr:
                print(f"  [Upload] stderr: {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [Upload] TIMEOUT after 2 hours")
        return False
    except FileNotFoundError:
        print(f"  [Upload] ERROR: 'dmf' command not found")
        return False


# =============================================================================
# Test Driver
# =============================================================================


def run_test(
    dataset_path: str,
    space_name: str,
    namespace: str,
    label: str,
    simulate_existing: bool,
    do_upload: bool,
    output_dir: str = None,
) -> bool:
    """
    Run the artifact upload test.

    Returns True if test passed, False otherwise.
    """
    print("=" * 60)
    print("Artifact Upload Test Driver")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Space: {space_name}")
    print(f"Namespace: {namespace}")
    print(f"Label: {label}")
    print(f"Simulate existing: {simulate_existing}")
    print(f"Upload enabled: {do_upload}")
    print()

    # Use dataset directory for output if not specified
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(dataset_path))

    # Initialize mock API
    api = MockRegistrationAPI()

    # Step 1: Calculate checksum
    print("Step 1: Calculating checksum...")
    print(f"  Using: {CHECKSUM_SCRIPT_PATH}")
    checksum, elapsed = calculate_checksum(dataset_path)

    if checksum is None:
        print("\n[ERROR] Checksum calculation failed")
        return False

    print(f"  Checksum: {checksum}")
    print()

    # Step 2: Simulate existing artifact if requested
    if simulate_existing:
        print("Step 2a: Pre-registering artifact (simulating existing)...")
        api.pre_register(checksum, space_name)
        print(f"  [Mock API] Pre-registered checksum {checksum[:16]}...")
        print()

    # Step 3: Check artifact registry
    print("Step 2: Checking artifact registry...")
    print(f"  [Mock API] Checking checksum {checksum[:16]}... in space {space_name}")
    response = api.check_artifact(checksum, space_name)

    status = response["status"]
    print(f"  Result: {status}")

    if status == "exists":
        artifact_uri = response["uri"]
        print(f"  Existing URI: {artifact_uri}")
        print(f"  Existing UUID: {response['uuid']}")
        print()

        # Step 3a: Generate build.yaml with existing artifact URI
        print("Step 3: Generating build.yaml with existing artifact URI...")
        build_yaml_path = generate_build_yaml(
            output_dir=output_dir,
            input_name=label,
            origin_path=dataset_path,
            checksum=checksum,
            lakehouse_uri=artifact_uri,
            job_id=f"test_{checksum[:8]}",
        )
        print(f"  Generated: {build_yaml_path}")
        print()
        print("  build.yaml contents:")
        with open(build_yaml_path, "r") as f:
            for line in f:
                print(f"    {line.rstrip()}")
        print()

        print("=" * 60)
        print("TEST RESULT: EXISTING ARTIFACT (upload skipped)")
        print(f"build.yaml: {build_yaml_path}")
        print("=" * 60)
        return True

    # Status is "not_found" - proceed to upload
    target_uri = response["target_uri"]
    print(f"  Target URI: {target_uri}")
    print()

    # Step 4: Register artifact
    print("Step 3: Registering artifact...")
    register_response = api.register_artifact(
        checksum=checksum,
        space_name=space_name,
        origin_uri=f"file://{os.path.abspath(dataset_path)}",
        namespace=namespace,
        label=label,
    )
    artifact_uri = register_response["uri"]
    print(f"  [Mock API] Registered: {artifact_uri}")
    print()

    # Step 4: Generate build.yaml with new artifact URI
    print("Step 4: Generating build.yaml...")
    build_yaml_path = generate_build_yaml(
        output_dir=output_dir,
        input_name=label,
        origin_path=dataset_path,
        checksum=checksum,
        lakehouse_uri=artifact_uri,
        job_id=f"test_{checksum[:8]}",
    )
    print(f"  Generated: {build_yaml_path}")
    print()
    print("  build.yaml contents:")
    with open(build_yaml_path, "r") as f:
        for line in f:
            print(f"    {line.rstrip()}")
    print()

    # Step 5: Upload artifact
    print("Step 5: Uploading artifact...")
    upload_success = run_dmf_upload(
        directory=dataset_path,
        namespace=namespace,
        label=label,
        version=checksum,
        dry_run=not do_upload,
    )
    print()

    # Summary
    print("=" * 60)
    if upload_success:
        if do_upload:
            print("TEST RESULT: NEW ARTIFACT (uploaded successfully)")
        else:
            print("TEST RESULT: NEW ARTIFACT (would upload)")
        print(f"build.yaml: {build_yaml_path}")
    else:
        print("TEST RESULT: UPLOAD FAILED")
    print("=" * 60)

    return upload_success


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test driver for artifact upload workflow with mock registration API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with existing dataset
  python test_registration_api_driver.py /path/to/dataset --space my_space

  # Test existing artifact case
  python test_registration_api_driver.py /path/to/dataset --space my_space --simulate-existing

  # Generate test data and run
  python test_registration_api_driver.py -gd /proj/granite-build/scratch

  # Actually run DMF upload
  python test_registration_api_driver.py /path/to/dataset --space my_space --upload
        """,
    )

    # Dataset path (positional or generated)
    parser.add_argument("dataset", nargs="?", help="Path to input dataset directory")

    # Data generation options
    parser.add_argument(
        "--generate",
        "-g",
        nargs=3,
        metavar=("NUM_FILES", "SIZE_MB", "DEPTH"),
        help="Generate test data: num_files size_mb depth",
    )
    parser.add_argument(
        "--generate-dir",
        "-gd",
        metavar="DIR",
        help="Base directory for generated test data",
    )

    # Artifact registration options
    parser.add_argument(
        "--space",
        "-s",
        default="test_space",
        help="Space name for artifact registration (default: test_space)",
    )
    parser.add_argument(
        "--namespace",
        "-n",
        default=DEFAULT_NAMESPACE,
        help=f"DMF namespace (default: {DEFAULT_NAMESPACE})",
    )
    parser.add_argument(
        "--label", "-l", help="DMF fileset label (default: derived from dataset path)"
    )

    # Test mode options
    parser.add_argument(
        "--simulate-existing",
        "-e",
        action="store_true",
        help="Pre-register checksum to simulate existing artifact",
    )
    parser.add_argument(
        "--upload",
        "-u",
        action="store_true",
        help="Actually run DMF upload (default: dry-run)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-o",
        metavar="DIR",
        help="Directory for build.yaml output (default: parent of dataset)",
    )

    # Cleanup
    parser.add_argument(
        "--keep",
        "-k",
        action="store_true",
        help="Keep generated test data (don't clean up)",
    )

    args = parser.parse_args()

    print(f"\nChecksum script: {CHECKSUM_SCRIPT_PATH}")

    # Determine dataset path
    generated_dir = None
    cleanup_dir = None

    # Default mode: generate small test dataset if no args
    use_defaults = not args.generate and not args.dataset and not args.generate_dir

    if args.generate or use_defaults or args.generate_dir:
        # Generate test data
        if not args.generate:
            # Default: 5 files, 50 MB, depth 0
            base_dir = args.generate_dir or "/tmp"
            print(f"[Default mode] Generating: 5 files, 50 MB in {base_dir}")
            num_files = 5
            size_mb = 50.0
            depth = 0
        else:
            num_files = int(args.generate[0])
            size_mb = float(args.generate[1])
            depth = int(args.generate[2])
            if not args.generate_dir:
                parser.error("--generate-dir (-gd) is required when using --generate")
            base_dir = args.generate_dir

        # Validate base directory exists
        if not os.path.isdir(base_dir):
            parser.error(f"Generate directory does not exist: {base_dir}")

        # Create temp directory for test data
        generated_dir = tempfile.mkdtemp(prefix="artifact_test_", dir=base_dir)
        if not args.keep:
            cleanup_dir = generated_dir

        print(f"\n[Setup] Generating test dataset in {generated_dir}")
        generate_test_dataset(num_files, size_mb, generated_dir, depth)

        dataset_path = generated_dir

    else:
        # Use provided dataset
        if not args.dataset:
            parser.error("Either provide dataset path or use --generate-dir (-gd)")

        dataset_path = args.dataset
        if not os.path.isdir(dataset_path):
            parser.error(f"Dataset directory does not exist: {dataset_path}")

    # Determine label
    label = args.label
    if not label:
        # Derive from dataset path
        label = Path(dataset_path).name
        if label.startswith("artifact_test_"):
            label = "test_input"

    # Determine output directory for build.yaml
    output_dir = args.output_dir
    if not output_dir:
        if generated_dir:
            output_dir = generated_dir
        else:
            output_dir = os.path.dirname(os.path.abspath(dataset_path))

    try:
        success = run_test(
            dataset_path=dataset_path,
            space_name=args.space,
            namespace=args.namespace,
            label=label,
            simulate_existing=args.simulate_existing,
            do_upload=args.upload,
            output_dir=output_dir,
        )

    finally:
        # Cleanup generated data
        if cleanup_dir and os.path.exists(cleanup_dir):
            print(f"\n[Cleanup] Removing generated test data: {cleanup_dir}")
            import shutil

            shutil.rmtree(cleanup_dir)
        elif generated_dir and args.keep:
            print(f"\n[Keep] Test data preserved at: {generated_dir}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
