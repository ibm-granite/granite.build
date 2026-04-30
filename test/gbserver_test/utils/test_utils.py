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

import asyncio
from pathlib import Path

from gbserver.types.errors import LogMonitoringFailedException
from gbserver.utils.unwrap_errors import unwrap_errors
from gbserver.utils.utils import get_common_ancestor, get_sha256sum


class TestSha256sum:
    def test_get_sha256sum(self):
        test_cases = [
            {
                "input": "abcdef",
                "expected": "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721",
            },
            {
                "input": b"abcdef",
                "expected": "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721",
            },
            {
                "input": b"0011",
                "expected": "a8d0b6f0939cfd883251f62b265f971ef8a5ab97eee32b91460f08b965601d93",
            },
            {
                "input": "😀 🎉",
                "expected": "79ec3d617be3e6eabbb2efc4896d7af174622b58540abe6931cb7479b3078d9e",
            },
            {
                "input": b"\xf0\x9f\x98\x80 \xf0\x9f\x8e\x89",
                "expected": "79ec3d617be3e6eabbb2efc4896d7af174622b58540abe6931cb7479b3078d9e",
            },
        ]
        for test_case in test_cases:
            test_input = test_case["input"]
            expected = test_case["expected"]
            actual = get_sha256sum(test_input)
            assert (
                actual == expected
            ), f"test_input: {test_input} actual: {actual} expected: {expected}"


class TestGetCommonAncestor:
    def test_get_common_ancestor(self):
        test_cases = [
            ["/a/b/c", "/a/b", "/a/b/c/d/e"],
            ["a/b/c"],
            ["a/b/c", "a/b"],
            ["a/b/c", "a/b", "a/b/c/d/e"],
            [
                "experiments/testavoidprclone2/build.yaml",
                "experiments/testavoidprclone2/foo.json",
                "experiments/testavoidprclone2/run.yaml",
            ],
            ["a/b/c", "a/b", ""],
            ["a/b/c", "a/b", "random"],
        ]
        test_cases_answers = [
            "/a/b",
            "a/b/c",
            "a/b",
            "a/b",
            "experiments/testavoidprclone2",
            "",
            "",
        ]
        for test_case, expected_answer in zip(test_cases, test_cases_answers):
            test_case_paths = [Path(t) for t in test_case]
            expected_answer_path = Path(expected_answer)
            answer = get_common_ancestor(test_case_paths)
            assert answer == expected_answer_path, f"{answer} == {expected_answer_path}"


class TestUnwrapErrors:
    def test_unwrap_errors_1(self):
        async def f15_k8s__654_watch_for_pods():
            raise LogMonitoringFailedException(
                "Appwrapper gbn6s7h1gs status Failed. Terminating log monitoring"
            )

        async def f14_k8s__365_monitor_log_monitor():
            tasks = [asyncio.create_task(f15_k8s__654_watch_for_pods())]
            await asyncio.gather(*tasks)

        async def f13_k8s__371_monitor_log_monitor():
            await f14_k8s__365_monitor_log_monitor()
            lmfe = ValueError("f13_k8s__371_monitor_log_monitor lmfe error")
            raise lmfe

        async def f12_taskgroups__164__aexit():  # et, exc)
            await f13_k8s__371_monitor_log_monitor()
            raise BaseExceptionGroup()

        async def f11_taskgroups__71___aexit__():
            return await f12_taskgroups__164__aexit()  # et, exc)

        async def f10_targetrun__61__run():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(f11_taskgroups__71___aexit__())

        async def f9_run__91_run():
            task = asyncio.create_task(f10_targetrun__61__run())
            await task

        async def f8_taskgroups__164_aexit():  # et, exc)
            await f9_run__91_run()
            raise BaseExceptionGroup()

        async def f7_taskgroups__71___aexit__():
            return await f8_taskgroups__164_aexit()  # et, exc)

        async def f6_buildrun__114__target_trigger():  # tg)
            async with asyncio.TaskGroup() as build_taskgroup:
                build_taskgroup.create_task(f7_taskgroups__71___aexit__())

        async def f5_buildrun__109__run():
            await f6_buildrun__114__target_trigger()  # tg)

        async def f4_taskgroups__164_aexit():  # et, exc):
            await f5_buildrun__109__run()
            raise BaseExceptionGroup()

        async def f3_taskgroups__71___aexit__():
            return await f4_taskgroups__164_aexit()  # et, exc)

        async def f2_build__80__run():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(f3_taskgroups__71___aexit__())

        async def f1_run__91_run():
            task = asyncio.create_task(f2_build__80__run())
            try:
                await task
            except ExceptionGroup as eg:
                readable_error = unwrap_errors(eg)
                assert (
                    readable_error
                    == "log monitoring failed (also failed to fetch build logs): Appwrapper gbn6s7h1gs status Failed. Terminating log monitoring"
                )

        asyncio.run(f1_run__91_run())

    def test_unwrap_errors_2(self):
        def f13_step__50___init__():
            files = []
            STEP_FILE_NAME = "step.yaml"
            stepasset_dir = (
                "/tmp/tmpewfzm5ry/c45bec7625f674a8248e51c9bcb151ae3e44ce911108fe83839b526df4575e96"
            )
            assert len(files) > 0, f"failed to find a {STEP_FILE_NAME} in {stepasset_dir}"

        def f12_targetstep__40___init__():
            # self.step = Step(stepuri=targetstep.step_uri, context=context)
            f13_step__50___init__()

        def f11_target__77_assimilate():
            # self.targetsteps : List[TargetStep]= [TargetStep(self.build_id, self.event_q, targetstep, self.name, self.environment, self.build_workspace_dir, self.dir, context=self.context) for targetstep in self.config.steps]
            xs = [f12_targetstep__40___init__() for i in range(1)]

        def f10_entity__44___init__():
            # self.assimilate()
            f11_target__77_assimilate()

        def f9_buildentity__48___init__():
            # super().__init__(type, config, dir, stored)
            f10_entity__44___init__()

        def f8_target__73___init__():
            # super().__init__(build_id, event_q, "target", config, build_workspace_dir, self.target_workspace_dir)
            f9_buildentity__48___init__()

        def f7_build__132_assimilate():
            # self.targets[target_name] = Target()
            f8_target__73___init__()

        def f6_build__149_assimilate():
            try:
                f7_build__132_assimilate()
            except Exception as e:
                raise ValueError(
                    "failed to load the target processdataset for build 2691cac1-7477-4768-8ce4-49a0c07c7e6d :"
                ) from e

        def f5_entity__44___init__():
            # self.assimilate()
            f6_build__149_assimilate()

        def f4_buildentity__48___init__():
            # super().__init__(type, config, dir, stored)
            f5_entity__44___init__()

        def f3_build__106___init__():
            # super().__init__()
            f4_buildentity__48___init__()

        def f2_build__122___init__():
            try:
                f3_build__106___init__()
            except Exception as e:
                raise ValueError(
                    "Build 2691cac1-7477-4768-8ce4-49a0c07c7e6d failed on creation"
                ) from e

        def f1_buildrunner__177___async_start_build():
            # build = Build()
            f2_build__122___init__()

        expected_err = """failed to find a step.yaml in /tmp/tmpewfzm5ry/c45bec7625f674a8248e51c9bcb151ae3e44ce911108fe83839b526df4575e96
assert 0 > 0
 +  where 0 = len([])"""
        try:
            f1_buildrunner__177___async_start_build()
        except Exception as e:
            readable_error = unwrap_errors(e)
            assert readable_error == expected_err

    def test_unwrap_errors_3_base_exception(self):
        def f2_build__122___init__():
            try:
                raise BaseException("This is a base exception!")
            except Exception as e:
                raise ValueError(
                    "Build 2691cac1-7477-4768-8ce4-49a0c07c7e6d failed on creation"
                ) from e

        def f1_buildrunner__177___async_start_build():
            # build = Build()
            f2_build__122___init__()

        expected_err = """This is a base exception!"""
        try:
            f1_buildrunner__177___async_start_build()
        # except Exception as e:
        except BaseException as e:
            readable_error = unwrap_errors(e)
            assert readable_error == expected_err

    def test_unwrap_errors_4_base_exception_group(self):
        def f2_build__122___init__():
            try:
                inner = [ValueError("dividing 1 by 0"), KeyError("mykey1")]
                raise BaseExceptionGroup("This is a base exception!", inner)
            except Exception as e:
                raise ValueError(
                    "Build 2691cac1-7477-4768-8ce4-49a0c07c7e6d failed on creation"
                ) from e

        def f1_buildrunner__177___async_start_build():
            # build = Build()
            f2_build__122___init__()

        expected_err = """value error: dividing 1 by 0\nkey error: 'mykey1'"""
        try:
            f1_buildrunner__177___async_start_build()
        # except Exception as e:
        except BaseException as e:
            readable_error = unwrap_errors(e)
            assert readable_error == expected_err
