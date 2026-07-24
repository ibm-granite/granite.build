def convert_path_to_lh_uri(path: str) -> str:
    """Convert a lakehouse filesystem path to an lh:// URI.

    Example:
        /gb-lakehouse-prod-read-only/filesets/granite_dot_build/public/shared/climate/20250906T064534/climate_train.jsonl
        → lh://prod/granite_dot_build.public/filesets/fileset_shared/climate/20250906T064534
    """
    parts = path.strip("/").split("/")

    # Extract env from "gb-lakehouse-{env}-read-only"
    mount = parts[0]  # e.g. gb-lakehouse-prod-read-only
    env = mount.removeprefix("gb-lakehouse-").removesuffix("-read-only")

    # parts[1] == "filesets"
    catalog = parts[2]  # e.g. granite_dot_build
    schema = parts[3]  # e.g. public
    fileset = parts[4]  # e.g. shared

    # Remaining path excluding the filename
    rest = "/".join(parts[5:-1])  # e.g. climate/20250906T064534

    uri = f"lh://{env}/{catalog}.{schema}/filesets/fileset_{fileset}"
    if rest:
        uri += f"/{rest}"

    return uri


path = (
    "/gb-lakehouse-prod-read-only/filesets/granite_dot_build/public/shared/climate/20250906T064534/climate_train.jsonl"
)
print(convert_path_to_lh_uri(path))
