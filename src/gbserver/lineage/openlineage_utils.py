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

from __future__ import annotations

import random
from typing import Tuple
from urllib.parse import urlparse

from gbcommon.uri.hf import HfURI
from gbcommon.utils.hf_utils import get_hf_artifact_uri

# fmt: off
_ADJECTIVES = [
    "admiring", "adoring", "agitated", "amazing", "angry", "awesome", "blissful", "bold",
    "brave", "clever", "cool", "dazzling", "eager", "elegant", "epic", "festive", "fierce",
    "friendly", "funny", "gallant", "gifted", "gracious", "happy", "hopeful", "hungry",
    "infallible", "inspiring", "invincible", "jolly", "jovial", "keen", "kind", "laughing",
    "loving", "lucid", "magical", "modest", "mystifying", "nifty", "nostalgic", "optimistic",
    "pedantic", "pensive", "practical", "quirky", "relaxed", "romantic", "serene", "silly",
    "sleepy", "sneaky", "stoic", "strange", "stupefied", "sweet", "tender", "thirsty",
    "trusting", "unruffled", "upbeat", "vibrant", "vigilant", "wild", "wizardly", "wonderful",
    "xenial", "youthful", "zealous",
]

_ANIMALS = [
    "albatross", "alligator", "antelope", "armadillo", "baboon", "badger", "bat", "bear",
    "beaver", "bison", "boar", "buffalo", "bull", "butterfly", "calf", "camel", "capybara",
    "chameleon", "cheetah", "chipmunk", "cobra", "coyote", "crane", "crow", "deer", "dingo",
    "dolphin", "donkey", "duck", "eagle", "elk", "falcon", "ferret", "finch", "flamingo",
    "fox", "frog", "gazelle", "gecko", "giraffe", "goat", "goose", "gorilla", "hawk",
    "hedgehog", "heron", "hippo", "horse", "hyena", "ibis", "iguana", "jaguar", "kangaroo",
    "koala", "lemur", "leopard", "lion", "lizard", "llama", "lynx", "meerkat", "mink",
    "moose", "narwhal", "newt", "otter", "owl", "panda", "panther", "parrot", "peacock",
    "pelican", "penguin", "porcupine", "puma", "quokka", "rabbit", "raccoon", "raven",
    "rhino", "salamander", "seal", "skunk", "sloth", "snake", "sparrow", "stork", "swan",
    "tiger", "toad", "tortoise", "toucan", "viper", "vulture", "walrus", "weasel", "wolf",
    "wolverine", "wombat", "yak", "zebra",
]
# fmt: on


def random_run_name() -> str:
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}-{random.randint(0, 999)}"


def get_huggingface_hub_url(artifact_type: str, repo_id: str) -> str:
    if artifact_type == "model":
        return f"https://huggingface.co/{repo_id}"
    elif artifact_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    elif artifact_type == "bucket":
        return f"https://huggingface.co/buckets/{repo_id}"
    else:
        raise ValueError(
            f"Invalid artifact_type: '{artifact_type}'. Use 'model', 'dataset', or 'bucket'"
        )


def parse_hf_url(url: str) -> Tuple[str, str, str]:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    if parts[0] == "datasets":
        artifact_type = "dataset"
        org, artifact_name = parts[1:3]
    elif parts[0] == "spaces":
        artifact_type = "space"
        org, artifact_name = parts[1:3]
    elif parts[0] == "buckets":
        artifact_type = "bucket"
        org, artifact_name = parts[1:3]
    else:
        artifact_type = "model"
        org, artifact_name = parts[0:2]

    return org, artifact_name, artifact_type


def parse_hf_uri(uri: str) -> Tuple[str, str, str]:
    hf_uri = HfURI.parse(uri)
    org = hf_uri.get_owner()
    name = hf_uri.get_repo()
    hf_type = hf_uri.get_hf_type()
    artifact_type = hf_type.value if hf_type else "model"
    return org, name, artifact_type
