from pathlib import Path

from gbcli.utils.gbconstants import getenv_boolean, GBSERVER_INSTANCE

LLMBSUB_OUTPUT_HEADER = ["ARTIFACT_TYPE", "ARTIFACT_NAME", "ARTIFACT_PATH"]
CONFIG_FILE_CONTENTS = "CONFIG_FILE_CONTENTS"
ARTIFACT_NAME_CHARACTERS = "a-zA-Z0-9_*"

# Input upload configuration
UPLOAD_BASE_PATH = "/proj/granite-build/llmb/upload"
LLMBSUM_SCRIPT_PATH = "/proj/granite-build/tools/dirsum_sorted_128.sh"
GBSERVER_ARTIFACT_STATUS_API = f"{GBSERVER_INSTANCE}/api/v1/artifact/status"
UPLOAD_CHECKSUM_CONCURRENCY = 8

# Feature flags for input registration and upload
# LLMBSUB_INPUT_REGISTRATION: Enable artifact registration (checksum + API call + metadata in build.yaml)
# LLMBSUB_INPUT_UPLOAD: Enable DMF upload after registration (requires registration to be enabled)
# LLMBSUB_BUILD_YAML_TO_STDOUT: Output generated build.yaml to stdout (useful with --dry-run)
REGISTRATION_FEATURE_FLAG = getenv_boolean("LLMBSUB_INPUT_REGISTRATION", False)
UPLOAD_FEATURE_FLAG = getenv_boolean("LLMBSUB_INPUT_UPLOAD", False)
BUILD_YAML_TO_STDOUT = getenv_boolean("LLMBSUB_BUILD_YAML_TO_STDOUT", False)

# Validate: upload requires registration - if upload is enabled, enable registration too
if UPLOAD_FEATURE_FLAG and not REGISTRATION_FEATURE_FLAG:
    import logging

    logging.getLogger(__name__).warning(
        "LLMBSUB_INPUT_UPLOAD requires LLMBSUB_INPUT_REGISTRATION, enabling registration"
    )
    REGISTRATION_FEATURE_FLAG = True

FEATURE_FLAGS = {
    "use_project_log_folder": getenv_boolean(
        "USE_PROJECT_LOG_FOLDER", True
    ),  # Default true
    "enable_input_registration": REGISTRATION_FEATURE_FLAG,
    "enable_input_upload": UPLOAD_FEATURE_FLAG,
    "build_yaml_to_stdout": BUILD_YAML_TO_STDOUT,
}
