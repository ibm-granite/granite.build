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

import pytest

from gbserver.utils.template import fill_objtemplate


class TestFillObjTemplate:
    """Tests for the fill_objtemplate function."""

    def test_1(self):
        test_cases = [
            {
                "description": "empty dict",
                "obj": {},
                "data": {},
                "strict": True,
                "expected_obj": {},
            },
            {
                "description": "empty data",
                "obj": {"k1": "v1"},
                "data": {},
                "strict": True,
                "expected_obj": {"k1": "v1"},
            },
            {
                "description": "simple string fill",
                "obj": {"k1": "this is a {{ mytype }} template"},
                "data": {"mytype": "test1"},
                "strict": True,
                "expected_obj": {"k1": "this is a test1 template"},
            },
            {
                "description": "key in template is missing in data",
                "obj": {"k1": "this is a {{ mytype }} template"},
                "data": {},
                "strict": True,
                "raises": "failed to use the data to fill the template:\nthis is a {{ mytype }} template",
                "raises_inner": "'mytype' is undefined",
            },
            {
                "description": "realistic valid output URI",
                "obj": {
                    "uri": "lh://{{ space.variables.DEFAULT_LH_ENVIRONMENT }}/{{ space.variables.DEFAULT_LH_NAMESPACE }}/models/{{ space.variables.DEFAULT_LH_MODEL_TABLE }}/gb_tuned_model_{{ run_metadata.targetsteprun_id | short_hash }}_{{ binding.path | path_basename }}",
                },
                "data": {
                    "space": {
                        "variables": {
                            "DEFAULT_LH_ENVIRONMENT": "prod",
                            "DEFAULT_LH_NAMESPACE": "granite_dot_build.public",
                            "DEFAULT_LH_MODEL_TABLE": "model_shared",
                        }
                    },
                    "run_metadata": {
                        "targetsteprun_id": "9a5e0004-8205-402a-aedc-a8b9c37d80cb"
                    },
                    "binding": {
                        "path": "/aaa/bbb/ccc/dddd/checkpoint-10",
                    },
                },
                "strict": True,
                "expected_obj": {
                    "uri": "lh://prod/granite_dot_build.public/models/model_shared/gb_tuned_model_zchlki1g_checkpoint-10",
                },
            },
            {
                "description": "referring to a non-existent space variable DEFAULT_LH_MODEL_TABLE",
                "obj": {
                    "uri": "lh://{{ space.variables.DEFAULT_LH_ENVIRONMENT }}/{{ space.variables.DEFAULT_LH_NAMESPACE }}/models/{{ space.variables.DEFAULT_LH_MODEL_TABLE }}/gb_tuned_model_{{ run_metadata.targetsteprun_id | short_hash }}_{{ binding.path | path_basename }}",
                },
                "data": {
                    "space": {
                        "variables": {
                            "DEFAULT_LH_ENVIRONMENT": "prod",
                            "DEFAULT_LH_NAMESPACE": "granite_dot_build.public",
                        }
                    },
                    "run_metadata": {"targetsteprun_id": "targetsteprun_id_xxxxxxx"},
                    "binding": {
                        "path": "/aaa/bbb/ccc/dddd/checkpoint-10",
                    },
                },
                "strict": True,
                "raises": "failed to use the data to fill the template:\nlh://{{ space.variables.DEFAULT_LH_ENVIRONMENT }}/{{ space.variables.DEFAULT_LH_NAMESPACE }}/models/{{ space.variables.DEFAULT_LH_MODEL_TABLE }}/gb_tuned_model_{{ run_metadata.targetsteprun_id | short_hash }}_{{ binding.path | path_basename }}",
                "raises_inner": "'dict object' has no attribute 'DEFAULT_LH_MODEL_TABLE'",
            },
        ]
        for test_case in test_cases:
            print("test_case", test_case)
            raises_e = test_case.get("raises", None)
            if raises_e:
                assert isinstance(raises_e, str)
                with pytest.raises(Exception) as exc_info:
                    filled_obj = fill_objtemplate(
                        obj=test_case["obj"],
                        data=test_case["data"],
                        strict=test_case["strict"],
                    )
                assert raises_e in str(exc_info.value)
                raises_inner = test_case.get("raises_inner", None)
                nested_e = exc_info.value.__cause__
                if raises_inner is None:
                    assert nested_e is None
                else:
                    assert raises_inner == str(nested_e)
            else:
                filled_obj = fill_objtemplate(
                    obj=test_case["obj"],
                    data=test_case["data"],
                    strict=test_case["strict"],
                )
                assert filled_obj == test_case["expected_obj"]
