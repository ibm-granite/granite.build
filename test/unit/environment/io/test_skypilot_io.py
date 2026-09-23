from gbserver.environment.io.base import EnvironmentIO
from gbserver.environment.io.descriptors import HfInputIO, HfOutputIO
from gbserver.environment.io.hf_shell import HFPUSH_UPLOAD_BODY
from gbserver.environment.io.skypilot import SkypilotIO


def test_environmentio_defaults_are_empty():
    class NoopIO(EnvironmentIO):
        pass

    io = NoopIO()
    assert io.render_prologue([]) == ""
    assert io.render_epilogue([], "cap") == ""


def test_render_prologue_emits_hf_download():
    sh = SkypilotIO().render_prologue(
        [
            HfInputIO(
                repo="ns/model",
                revision="v1",
                type="model",
                dest="/cache/ns/model/v1",
                token="tok",
            )
        ]
    )
    assert 'hf download "ns/model" --local-dir "/cache/ns/model/v1"' in sh
    assert '--revision "v1"' in sh
    assert "--repo-type model" in sh


def test_render_prologue_empty_for_no_inputs():
    assert SkypilotIO().render_prologue([]) == ""


def test_render_epilogue_resolves_src_from_marker_and_uploads():
    out = HfOutputIO(
        repo="ns/out",
        revision="main",
        private=False,
        resource_group_id=None,
        path_in_repo="",
        uri="hf://ns/out",
        binding_id="model_out",
        token="tok",
        hf_type="model",
    )
    sh = SkypilotIO().render_epilogue([out], capture_var="GB_INLINE_PUSH_CAP")

    # Resolves HF_SOURCE from the tee'd capture for the right binding_id.
    assert "GB_INLINE_PUSH_CAP" in sh
    assert "model_out" in sh
    assert "GB_ARTIFACT_PATH" in sh
    # Matches either GB_ or LLMB_ prefix on the marker.
    assert "LLMB_" in sh
    # Fail-fast when the marker is missing.
    assert "exit 1" in sh
    # Uses the shared upload body and emits the pushed marker.
    assert 'echo "Pushed HF URI: ${HF_URI} for binding ${BINDING_ID}"' in sh
    assert HFPUSH_UPLOAD_BODY in sh
    assert "HF_URI='hf://ns/out'" in sh or 'HF_URI="hf://ns/out"' in sh
    # Every var HFPUSH_UPLOAD_BODY reads is set by the epilogue.
    for var in (
        "HF_SOURCE",
        "HF_URI",
        "HF_ENDPOINT",
        "HF_OWNER",
        "HF_REPO_NAME",
        "HF_REPO",
        "HF_REVISION",
        "HF_PATH_IN_REPO",
        "HF_PRIVATE",
        "HF_TYPE",
        "HF_RESOURCE_GROUP_ID",
        "BINDING_ID",
    ):
        assert var in sh


def test_render_epilogue_sets_owner_and_repo_name_from_repo():
    out = HfOutputIO(
        repo="acme/widgets",
        uri="hf://acme/widgets",
        binding_id="b1",
    )
    sh = SkypilotIO().render_epilogue([out], capture_var="CAP")
    assert "HF_OWNER='acme'" in sh
    assert "HF_REPO_NAME='widgets'" in sh


def test_render_epilogue_shell_quotes_dangerous_values():
    out = HfOutputIO(
        repo="ns/out",
        uri="hf://ns/out",
        binding_id="b'; rm -rf /",
    )
    sh = SkypilotIO().render_epilogue([out], capture_var="CAP")
    # Single-quote escaping keeps the injection inert.
    assert "'; rm -rf /" not in sh.replace("'\\''", "")
    assert "BINDING_ID='b'\\''; rm -rf /'" in sh


def test_render_epilogue_empty_for_no_outputs():
    assert SkypilotIO().render_epilogue([], "cap") == ""
