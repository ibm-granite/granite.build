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

"""Build-time security patch for transformers 4.55.4: CVE-2026-5241.

Also listed as GHSA-fgcw-684q-jj6r and PYSEC-2026-2290; quay's scanner rates it
Critical. LightGlueConfig takes ``trust_remote_code`` from the model's own
config.json and passes it to a nested ``AutoConfig.from_pretrained`` for a
keypoint detector that is not a built-in model type. So loading an
attacker-controlled model repo runs its Python, even when the caller never
allowed remote code. fms-hf-tuning loads ``base_model`` with
``AutoModelForCausalLM.from_pretrained`` and no ``trust_remote_code``, and is
still exposed.

fms-hf-tuning 3.2.2 caps ``transformers<=4.55.4``, so the image cannot take the
fixed release. This backports upstream's fix, huggingface/transformers 676559d5
("[LightGlue] Remove remote code execution", #45122), to 4.55.4's code. The
keypoint detector config is always built from ``CONFIG_MAPPING`` (an unknown
type raises KeyError), and the detector model never gets ``trust_remote_code``.
LightGlue is a keypoint matcher, so nothing the step trains uses it.

Exits non-zero if any target is not found exactly once, so a transformers bump
fails the image build instead of silently shipping unpatched code.
"""

import pathlib
import sys

EDITS = {
    "configuration_lightglue.py": (
        '            if keypoint_detector_config["model_type"] not in CONFIG_MAPPING:\n'
        "                keypoint_detector_config = AutoConfig.from_pretrained(\n"
        '                    keypoint_detector_config["_name_or_path"], trust_remote_code=self.trust_remote_code\n'
        "                )\n"
        "            else:\n"
        '                keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](\n'
        '                    **keypoint_detector_config, attn_implementation="eager"\n'
        "                )\n",
        '            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](\n'
        '                **keypoint_detector_config, attn_implementation="eager"\n'
        "            )\n",
    ),
    "modeling_lightglue.py": (
        "        self.keypoint_detector = AutoModelForKeypointDetection.from_config(\n"
        "            config.keypoint_detector_config, trust_remote_code=config.trust_remote_code\n"
        "        )\n",
        "        self.keypoint_detector = AutoModelForKeypointDetection.from_config(config.keypoint_detector_config)\n",
    ),
}


def patch(lightglue_dir: pathlib.Path) -> None:
    """Apply the backport to the files in ``lightglue_dir`` in place.

    :param lightglue_dir: transformers/models/lightglue inside the venv.
    :raises SystemExit: If a target is not found exactly once in its file.
    """
    texts = {}
    for name, (old, new) in EDITS.items():
        path = lightglue_dir / name
        text = path.read_text()
        if text.count(old) != 1:
            sys.exit(
                f"fms-hf-tuning: CVE-2026-5241 patch target not found exactly once in {path}"
            )
        texts[path] = text.replace(old, new)
    for path, text in texts.items():
        path.write_text(text)


if __name__ == "__main__":
    patch(pathlib.Path(sys.argv[1]))
