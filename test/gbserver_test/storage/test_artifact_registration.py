from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.types.artifact import ArtifactType


class TestArtifactRegistration:
    def _test_tags(self, tags: list[str], expect_fail: bool) -> None:
        try:
            ArtifactRegistration(
                type=ArtifactType.MODEL,
                uri="https://foo.bar",
                username="me",
                space_name="space",
                tags=tags,
            )
            got_exception_on_init = False
        except Exception as e:
            got_exception_on_init = True

        try:
            artifact = ArtifactRegistration(
                type=ArtifactType.MODEL,
                uri="https://foo.bar",
                username="me",
                space_name="space",
            )
            artifact.tags = tags
            got_exception_on_set = False
        except Exception as e:
            got_exception_on_set = True

        assert (
            got_exception_on_init == expect_fail
        ), "Did not get an exception as expected during initialization"
        assert (
            got_exception_on_set == expect_fail
        ), "Did not get an exception as expected during assignment operation"

    def test_tag_validation(self) -> None:
        self._test_tags([], False)
        self._test_tags(["a"], False)
        self._test_tags(["a", "b"], False)
        self._test_tags(["a "], True)
        self._test_tags(["a,"], True)
        self._test_tags(["a\t"], True)
        self._test_tags(["a\n"], True)
        self._test_tags(["a\r"], True)
        self._test_tags(["a", " b "], True)
        self._test_tags(["a", " b "], True)
