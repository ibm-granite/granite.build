"""Lh model module."""

from pathlib import Path

from gbcli.utils.gbconstants import LAKEHOUSE_MODEL_SHARED_TABLE


def createModel(
    lh,
    path_name: Path,
    namespace: str,
    table_name: str,
    model_label: str,
    size: str,
    variant: str,
    model_type: str,
    revision: str,
    disable_aspera: bool,
):
    """Create model."""
    try:
        from lakehouse.assets import Model

        if getModel(lh, namespace, table_name, model_label, revision):
            raise Exception(
                f"Error: A Lakehouse model with the label '{model_label}' and revision '{revision}' already exists already exists in namespace '{namespace}'. Please choose a different label or revision."
            )

        model_dir = None
        filename = None
        if path_name.is_dir():
            model_dir = path_name.resolve().as_posix()
        elif path_name.is_file():
            if path_name.suffix != "yaml":
                raise Exception(
                    "File not supported. Provide a path to model checkpoint metadata yaml"
                )
            else:
                filename = path_name.resolve().as_posix()

        model = Model(lh=lh)

        model_team = model.push(
            model_dir=model_dir,
            filename=filename,
            label=model_label,
            table=table_name,
            namespace=namespace,
            overwrite=True,
            size=size,
            revision=revision,
            variant=variant,
            type=model_type,
            use_aspera=(not disable_aspera),
            open=(table_name == LAKEHOUSE_MODEL_SHARED_TABLE),
        )

        if not (isinstance(model_team, str) and model_team.startswith("ERROR:")):
            return model_team

        raise Exception(f"Error: Pushing Model : '{model_team}'")

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e


def pullModel(
    lh,
    model_dir: Path,
    namespace: str,
    model_label: str,
    table_name: str,
    revision: str,
    disable_aspera: str,
):
    """Pull model."""
    try:
        from lakehouse.assets import Model

        model = Model(lh=lh)
        model.pull(
            model_label,
            revision,
            namespace,
            table_name,
            model_dir,
            use_aspera=(not disable_aspera),
        )

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e


def getModel(
    lh,
    namespace: str,
    table_name: str,
    model_label: str,
    revision: str,
):
    """Get the model."""
    try:
        from lakehouse.assets import Model

        model = Model(lh=lh)

        return model._get_model(model_label, revision, namespace, table_name, ("*"))

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e


def get_model_subforlder(model, revision):
    """Get the model subforlder."""
    return f"{model}/{revision}" if revision != "-1" else model


def copyModel(
    lh,
    source_namespace: str,
    target_namespace: str,
    source_table: str,
    target_table: str,
    model_lable: str,
    revision: str,
    open: bool,
):
    """Copy model."""
    try:
        from lakehouse.assets import Model
        from lakehouse.core import CopyAssetStatus

        model = Model(lh=lh)
        # Copy a restricted model to the model_shared table with open access.
        status: CopyAssetStatus = model.copy_model(
            source_namespace=source_namespace,
            source_table=source_table,
            target_namespace=target_namespace,
            target_table=target_table,
            model=model_lable,
            revision=revision,
            open=open,
        )
        if not (isinstance(status, str) and status.startswith("ERROR:")):
            return status

        raise Exception(f"Error: Copying Model : '{status}'")

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e
