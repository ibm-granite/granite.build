"""Lh dataset module."""


def dataset_info_lh(lh, dataset_name: str, namespace: str):
    """Dataset info lh."""
    try:

        from lakehouse.assets.dataset import Dataset  # type: ignore

        dataset_info = Dataset(lh=lh, dataset_name=dataset_name, namespace=namespace)
        return dataset_info

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )
