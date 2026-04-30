"""Lh fileset module."""

from pathlib import Path

from gbcli.utils.utils import retry_function


def createFileset(
    lh,
    path_name: str,
    namespace: str,
    file_label: str,
    version: str,
    table_name: str,
    disable_aspera: bool,
):
    """Create fileset."""
    try:

        from lakehouse.assets.fileset import Fileset

        fileset = Fileset(lh=lh, namespace=namespace, table=table_name, create_if_not_exists=True)

        local_path = Path(path_name)
        if local_path.is_file():
            result = fileset.push(
                label=file_label,
                version=version,
                file_location=path_name,
                use_aspera=(not disable_aspera),
            )
        else:
            result = fileset.push(
                label=file_label,
                version=version,
                dir=path_name,
                use_aspera=(not disable_aspera),
            )

        if result and not (isinstance(result, str) and result.startswith("ERROR:")):
            return result

        raise Exception(f"Error: Pushing Fileset : '{result}'")

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e


def pullFileset(
    lh,
    path_name: Path,
    namespace: str,
    file_label: str,
    table_name: str,
    version: str,
    disable_aspera: bool,
):
    """Pull fileset."""
    try:
        from lakehouse.assets.fileset import Fileset

        fileset = Fileset(
            lh=lh,
            namespace=namespace,
            table=table_name,
            create_if_not_exists=True,
        )
        retry_function(fileset.pull, 5, 1, file_label, version, path_name, True, not disable_aspera)

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e


def checkFileset(
    lh,
    namespace: str,
    file_label: str,
    table_name: str,
    version: str,
):
    """Verify fileset."""
    try:
        from lakehouse.assets.fileset import Fileset

        fileset = Fileset(lh=lh, namespace=namespace, table=table_name, create_if_not_exists=True)
        flist = fileset.list(file_label, version, include_files=False)

        cur = {"label": file_label, "version": version}

        if cur in flist:
            return True
        else:
            return False

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e


def get_fileset_subforlder(fileset_label, fileset_version):
    """Get the fileset subforlder."""
    return f"{fileset_label}/{fileset_version}"
