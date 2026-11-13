# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def dataset_info_lh(lh, dataset_name: str, namespace: str):
    try:

        from lakehouse.assets.dataset import Dataset  # type: ignore

        dataset_info = Dataset(lh=lh, dataset_name=dataset_name, namespace=namespace)
        return dataset_info

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )
