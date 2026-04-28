#!/usr/bin/env python3

# Copyright LLM.build Authors
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

import re

import pytest

from gbserver.lineage.openlineage_utils import (
    get_hf_artifact_uri,
    get_huggingface_hub_url,
    parse_hf_uri,
    parse_hf_url,
    random_run_name,
)

pytestmark = pytest.mark.g4os


class TestRandomRunName:
    def test_format(self):
        name = random_run_name()
        assert re.match(r"^[a-z]+-[a-z]+-\d+$", name), f"Unexpected format: {name}"

    def test_uniqueness(self):
        names = {random_run_name() for _ in range(50)}
        assert len(names) > 1


class TestGetHuggingfaceHubUrl:
    def test_model(self):
        url = get_huggingface_hub_url("model", "org/my-model")
        assert url == "https://huggingface.co/org/my-model"

    def test_dataset(self):
        url = get_huggingface_hub_url("dataset", "org/my-dataset")
        assert url == "https://huggingface.co/datasets/org/my-dataset"

    def test_bucket(self):
        url = get_huggingface_hub_url("bucket", "org/my-bucket")
        assert url == "https://huggingface.co/buckets/org/my-bucket"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid artifact_type"):
            get_huggingface_hub_url("space", "org/thing")


class TestGetHfArtifactUri:
    def test_model(self):
        uri = get_hf_artifact_uri("org/my-model", "model")
        assert uri == "hf:///models/org/my-model"

    def test_dataset(self):
        uri = get_hf_artifact_uri("org/my-dataset", "dataset")
        assert uri == "hf:///datasets/org/my-dataset"

    def test_bucket(self):
        uri = get_hf_artifact_uri("org/my-bucket", "bucket")
        assert uri == "hf:///buckets/org/my-bucket"


class TestParseHfUrl:
    def test_model_url(self):
        org, name, atype = parse_hf_url("https://huggingface.co/org/my-model")
        assert org == "org"
        assert name == "my-model"
        assert atype == "model"

    def test_dataset_url(self):
        org, name, atype = parse_hf_url(
            "https://huggingface.co/datasets/org/my-dataset"
        )
        assert org == "org"
        assert name == "my-dataset"
        assert atype == "dataset"

    def test_space_url(self):
        org, name, atype = parse_hf_url("https://huggingface.co/spaces/org/my-space")
        assert org == "org"
        assert name == "my-space"
        assert atype == "space"

    def test_bucket_url(self):
        org, name, atype = parse_hf_url("https://huggingface.co/buckets/org/my-bucket")
        assert org == "org"
        assert name == "my-bucket"
        assert atype == "bucket"


class TestParseHfUri:
    def test_model_uri(self):
        org, name, atype = parse_hf_uri("hf:///models/org/my-model")
        assert org == "org"
        assert name == "my-model"
        assert atype == "model"

    def test_dataset_uri(self):
        org, name, atype = parse_hf_uri("hf:///datasets/org/my-dataset")
        assert org == "org"
        assert name == "my-dataset"
        assert atype == "dataset"

    def test_bucket_uri(self):
        org, name, atype = parse_hf_uri("hf:///buckets/org/my-bucket")
        assert org == "org"
        assert name == "my-bucket"
        assert atype == "bucket"
