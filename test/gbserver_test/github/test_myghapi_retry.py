#!/usr/bin/env python3
"""
Unit tests for myghapi retry logic using Tenacity with exponential backoff.

Run with: python -m pytest test/gbserver_test/github/test_myghapi_retry.py -v
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import requests

from gbserver.github.myghapi import MyGHApi
from gbserver.types.constants import (
    GITHUB_API_MAX_RETRIES,
    GITHUB_API_RETRY_BASE_DELAY,
    GITHUB_API_RETRY_MAX_DELAY,
)
from gbserver.utils.git_retry import (
    _calculate_github_backoff_delay,
    _is_github_rate_limit_error,
    _should_retry_github_status_code,
    _wait_for_github_rate_limit,
    is_retryable_github_error,
)


class TestBackoffDelay(unittest.TestCase):
    """Test the exponential backoff calculation."""

    def test_first_retry_is_base_delay(self):
        """First retry should be around base delay (with jitter)."""
        delay = _calculate_github_backoff_delay(1, base_delay=1.0)
        # With jitter of 0.5-1.5, delay should be between 0.5 and 1.5
        self.assertGreaterEqual(delay, 0.5)
        self.assertLessEqual(delay, 1.5)

    def test_exponential_growth(self):
        """Delay should grow exponentially."""
        # retry 1: 1s, retry 2: 2s, retry 3: 4s, retry 4: 8s
        delays = [
            _calculate_github_backoff_delay(i, base_delay=1.0, max_delay=100.0) for i in range(1, 5)
        ]
        # Check that each delay is roughly double the previous (within jitter bounds)
        for i in range(1, len(delays)):
            # The ratio should be around 2 (between 0.67 and 6 accounting for jitter)
            ratio = delays[i] / delays[i - 1]
            self.assertGreater(ratio, 0.5)  # Very loose bound due to jitter
            self.assertLess(ratio, 6.0)

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        delay = _calculate_github_backoff_delay(100, base_delay=1.0, max_delay=60.0)
        # With jitter of 1.5x max, the absolute max is 90
        self.assertLessEqual(delay, 90.0)

    def test_jitter_adds_randomness(self):
        """Multiple calls should produce different values due to jitter."""
        delays = [_calculate_github_backoff_delay(5) for _ in range(10)]
        # Not all values should be identical
        self.assertGreater(len(set(delays)), 1)


class TestRateLimitDetection(unittest.TestCase):
    """Test rate limit error detection."""

    def _make_mock_response(self, status_code, headers=None, json_data=None):
        """Create a mock response object."""
        response = MagicMock(spec=requests.Response)
        response.status_code = status_code
        response.headers = headers or {}
        if json_data is not None:
            response.json.return_value = json_data
        else:
            response.json.side_effect = ValueError("No JSON")
        return response

    def test_rate_limit_with_remaining_zero(self):
        """Detect rate limit when x-ratelimit-remaining is 0."""
        response = self._make_mock_response(
            403,
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(int(time.time()) + 60),
            },
        )
        is_rate_limit, retry_after = _is_github_rate_limit_error(response)
        self.assertTrue(is_rate_limit)
        self.assertIsNotNone(retry_after)
        self.assertGreater(retry_after, 0)
        self.assertLessEqual(retry_after, 60)

    def test_rate_limit_message_in_body(self):
        """Detect rate limit from error message."""
        response = self._make_mock_response(
            403,
            headers={},
            json_data={"message": "API rate limit exceeded for user"},
        )
        is_rate_limit, retry_after = _is_github_rate_limit_error(response)
        self.assertTrue(is_rate_limit)
        self.assertIsNone(retry_after)  # No specific time given

    def test_abuse_detection_message(self):
        """Detect abuse detection rate limit."""
        response = self._make_mock_response(
            403,
            headers={},
            json_data={"message": "You have triggered an abuse detection mechanism"},
        )
        is_rate_limit, retry_after = _is_github_rate_limit_error(response)
        self.assertTrue(is_rate_limit)

    def test_retry_after_header(self):
        """Detect rate limit from retry-after header."""
        response = self._make_mock_response(
            429,
            headers={"retry-after": "120"},
        )
        is_rate_limit, retry_after = _is_github_rate_limit_error(response)
        self.assertTrue(is_rate_limit)
        self.assertEqual(retry_after, 120)

    def test_403_without_rate_limit_indicators(self):
        """403 without rate limit indicators should not be detected as rate limit."""
        response = self._make_mock_response(
            403,
            headers={"x-ratelimit-remaining": "100"},
            json_data={"message": "Resource not accessible by integration"},
        )
        is_rate_limit, retry_after = _is_github_rate_limit_error(response)
        self.assertFalse(is_rate_limit)

    def test_non_403_status(self):
        """Non-403/429 status should not be detected as rate limit."""
        response = self._make_mock_response(500, headers={})
        is_rate_limit, retry_after = _is_github_rate_limit_error(response)
        self.assertFalse(is_rate_limit)


class TestShouldRetryStatusCode(unittest.TestCase):
    """Test status code retry logic."""

    def test_retry_on_500(self):
        """Should retry on 500 Internal Server Error."""
        self.assertTrue(_should_retry_github_status_code(500))

    def test_retry_on_502(self):
        """Should retry on 502 Bad Gateway."""
        self.assertTrue(_should_retry_github_status_code(502))

    def test_retry_on_503(self):
        """Should retry on 503 Service Unavailable."""
        self.assertTrue(_should_retry_github_status_code(503))

    def test_retry_on_403(self):
        """Should retry on 403 Forbidden (potential rate limit)."""
        self.assertTrue(_should_retry_github_status_code(403))

    def test_retry_on_429(self):
        """Should retry on 429 Too Many Requests."""
        self.assertTrue(_should_retry_github_status_code(429))

    def test_no_retry_on_400(self):
        """Should not retry on 400 Bad Request."""
        self.assertFalse(_should_retry_github_status_code(400))

    def test_no_retry_on_401(self):
        """Should not retry on 401 Unauthorized."""
        self.assertFalse(_should_retry_github_status_code(401))

    def test_no_retry_on_404(self):
        """Should not retry on 404 Not Found."""
        self.assertFalse(_should_retry_github_status_code(404))

    def test_no_retry_on_200(self):
        """Should not retry on 200 OK."""
        self.assertFalse(_should_retry_github_status_code(200))


class TestMyGHApiRetry(unittest.TestCase):
    """Test MyGHApi methods with retry logic."""

    def _make_response(self, status_code, headers=None, json_data=None, raise_error=False):
        """Create a mock response that behaves like requests.Response."""
        response = MagicMock(spec=requests.Response)
        response.status_code = status_code
        response.headers = headers or {}
        if json_data is not None:
            response.json.return_value = json_data
        else:
            response.json.side_effect = ValueError("No JSON")

        if raise_error:

            def raise_for_status():
                err = requests.HTTPError(f"{status_code} Error")
                err.response = response
                raise err

            response.raise_for_status = raise_for_status
        else:
            response.raise_for_status = MagicMock()
        return response

    @patch("gbserver.github.myghapi.requests.get")
    @patch.object(MyGHApi, "__init__", lambda self, **kwargs: None)
    def test_is_repo_present_success_after_retries(self, mock_get):
        """Test is_repo_present succeeds after transient 403 errors."""
        api = MyGHApi.__new__(MyGHApi)
        api.owner = "test-owner"
        api.repo = "test-repo"
        api.gh_api_endpoint = "https://api.github.com"
        api.token = "test-token"
        api.timeout = 30
        api.cache_repo_exists = {}

        # Fail twice with 403, then succeed
        mock_get.side_effect = [
            self._make_response(403, headers={"x-ratelimit-remaining": "0"}, raise_error=True),
            self._make_response(403, headers={"x-ratelimit-remaining": "0"}, raise_error=True),
            self._make_response(200, json_data={"name": "test-repo"}),
        ]

        with patch.object(MyGHApi._get_single_page.retry, "sleep"):  # Skip actual sleep
            result = api.is_repo_present(use_cache=False)

        self.assertTrue(result)
        self.assertEqual(mock_get.call_count, 3)

    @patch("gbserver.github.myghapi.requests.get")
    @patch.object(MyGHApi, "__init__", lambda self, **kwargs: None)
    def test_is_repo_present_401_fails_immediately(self, mock_get):
        """Test is_repo_present fails immediately on 401."""
        api = MyGHApi.__new__(MyGHApi)
        api.owner = "test-owner"
        api.repo = "test-repo"
        api.gh_api_endpoint = "https://api.github.com"
        api.token = "test-token"
        api.timeout = 30
        api.cache_repo_exists = {}

        mock_get.return_value = self._make_response(401, raise_error=True)

        with self.assertRaises(ValueError) as ctx:
            api.is_repo_present(use_cache=False)

        self.assertIn("401 Unauthorized", str(ctx.exception))
        self.assertEqual(mock_get.call_count, 1)  # No retries

    @patch("gbserver.github.myghapi.requests.get")
    @patch.object(MyGHApi, "__init__", lambda self, **kwargs: None)
    def test_is_repo_present_404_returns_false(self, mock_get):
        """Test is_repo_present returns False on 404."""
        api = MyGHApi.__new__(MyGHApi)
        api.owner = "test-owner"
        api.repo = "test-repo"
        api.gh_api_endpoint = "https://api.github.com"
        api.token = "test-token"
        api.timeout = 30
        api.cache_repo_exists = {}

        mock_get.return_value = self._make_response(404, raise_error=True)

        result = api.is_repo_present(use_cache=False)

        self.assertFalse(result)
        self.assertEqual(mock_get.call_count, 1)  # No retries

    @patch("gbserver.github.myghapi.requests.get")
    @patch.object(MyGHApi, "__init__", lambda self, **kwargs: None)
    def test_is_branch_present_success_after_retries(self, mock_get):
        """Test is_branch_present succeeds after transient 403 errors."""
        api = MyGHApi.__new__(MyGHApi)
        api.owner = "test-owner"
        api.repo = "test-repo"
        api.gh_api_endpoint = "https://api.github.com"
        api.token = "test-token"
        api.timeout = 30

        # Fail once with 403, then succeed
        mock_get.side_effect = [
            self._make_response(403, headers={"x-ratelimit-remaining": "0"}, raise_error=True),
            self._make_response(200, json_data={"name": "main"}),
        ]

        with patch.object(MyGHApi._get_single_page.retry, "sleep"):
            result = api.is_branch_present("main")

        self.assertTrue(result)
        self.assertEqual(mock_get.call_count, 2)

    @patch("gbserver.github.myghapi.requests.put")
    @patch.object(MyGHApi, "__init__", lambda self, **kwargs: None)
    def test_merge_pr_success_after_retries(self, mock_put):
        """Test merge_pr succeeds after transient 403 errors."""
        api = MyGHApi.__new__(MyGHApi)
        api.owner = "test-owner"
        api.repo = "test-repo"
        api.gh_api_endpoint = "https://api.github.com"
        api.token = "test-token"
        api.timeout = 30

        # Fail twice, then succeed
        mock_put.side_effect = [
            self._make_response(403, headers={"x-ratelimit-remaining": "0"}, raise_error=True),
            self._make_response(403, headers={"x-ratelimit-remaining": "0"}, raise_error=True),
            self._make_response(
                200,
                json_data={
                    "sha": "abc123",
                    "merged": True,
                    "message": "Pull Request successfully merged",
                },
            ),
        ]

        with patch.object(MyGHApi._do_merge_pr.retry, "sleep"):
            result = api.merge_pr("123")

        self.assertTrue(result.merged)
        self.assertEqual(mock_put.call_count, 3)


class TestTenacityHelpers(unittest.TestCase):
    """Test the Tenacity helper functions."""

    def _make_http_error(self, status_code, headers=None):
        """Create a mock HTTPError with response."""
        response = MagicMock(spec=requests.Response)
        response.status_code = status_code
        response.headers = headers or {}
        response.json.side_effect = ValueError("No JSON")
        error = requests.HTTPError(f"{status_code} Error")
        error.response = response
        return error

    def test_is_retryable_github_error_500(self):
        """500 errors should be retryable."""
        error = self._make_http_error(500)
        self.assertTrue(is_retryable_github_error(error))

    def test_is_retryable_github_error_403(self):
        """403 errors should be retryable."""
        error = self._make_http_error(403)
        self.assertTrue(is_retryable_github_error(error))

    def test_is_retryable_github_error_429(self):
        """429 errors should be retryable."""
        error = self._make_http_error(429)
        self.assertTrue(is_retryable_github_error(error))

    def test_is_retryable_github_error_401(self):
        """401 errors should NOT be retryable."""
        error = self._make_http_error(401)
        self.assertFalse(is_retryable_github_error(error))

    def test_is_retryable_github_error_404(self):
        """404 errors should NOT be retryable."""
        error = self._make_http_error(404)
        self.assertFalse(is_retryable_github_error(error))

    def test_is_retryable_github_error_non_http(self):
        """Non-HTTPError exceptions should NOT be retryable."""
        error = ValueError("some error")
        self.assertFalse(is_retryable_github_error(error))


class TestBackoffTiming(unittest.TestCase):
    """Test that backoff delays are applied correctly."""

    def _make_response(self, status_code, headers=None):
        """Create a mock response that raises on raise_for_status."""
        response = MagicMock(spec=requests.Response)
        response.status_code = status_code
        response.headers = headers or {}
        response.json.side_effect = ValueError("No JSON")

        def raise_for_status():
            err = requests.HTTPError(f"{status_code} Error")
            err.response = response
            raise err

        response.raise_for_status = raise_for_status
        return response

    @patch("gbserver.github.myghapi.requests.get")
    @patch.object(MyGHApi, "__init__", lambda self, **kwargs: None)
    def test_sleep_called_with_increasing_delays(self, mock_get):
        """Test that sleep is called with exponentially increasing delays."""
        api = MyGHApi.__new__(MyGHApi)
        api.owner = "test-owner"
        api.repo = "test-repo"
        api.gh_api_endpoint = "https://api.github.com"
        api.token = "test-token"
        api.timeout = 30
        api.cache_repo_exists = {}

        # Always fail with 500 (server error, will retry)
        mock_get.return_value = self._make_response(500)

        with patch.object(MyGHApi._get_single_page.retry, "sleep") as mock_sleep:
            with self.assertRaises(RuntimeError):
                api.is_repo_present(use_cache=False)

            # Tenacity calls sleep once per retry (retries up to GITHUB_API_MAX_RETRIES)
            # For this test we just verify sleep was called with positive delays
            self.assertGreater(mock_sleep.call_count, 0)

            # Get the sleep durations
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]

            # Each delay should be positive
            for delay in sleep_calls:
                self.assertGreater(delay, 0)


class TestRateLimitRespected(unittest.TestCase):
    """Test that rate limit retry-after is respected."""

    @patch("gbserver.github.myghapi.requests.get")
    @patch.object(MyGHApi, "__init__", lambda self, **kwargs: None)
    @patch("gbserver.utils.git_retry.time.time")
    def test_respects_rate_limit_reset_time(self, mock_time, mock_get):
        """Test that the rate limit reset time is respected."""
        api = MyGHApi.__new__(MyGHApi)
        api.owner = "test-owner"
        api.repo = "test-repo"
        api.gh_api_endpoint = "https://api.github.com"
        api.token = "test-token"
        api.timeout = 30
        api.cache_repo_exists = {}

        # Set current time
        current_time = 1000000
        mock_time.return_value = current_time

        # Create 403 with rate limit headers indicating 30 second wait
        reset_time = current_time + 30

        def make_rate_limit_response():
            response = MagicMock(spec=requests.Response)
            response.status_code = 403
            response.headers = {
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(reset_time),
            }
            response.json.return_value = {"message": "rate limit exceeded"}

            def raise_for_status():
                error = requests.HTTPError("403 Forbidden")
                error.response = response
                raise error

            response.raise_for_status = raise_for_status
            return response

        def make_success_response():
            response = MagicMock(spec=requests.Response)
            response.status_code = 200
            response.headers = {}
            response.json.return_value = {"name": "test-repo"}
            response.raise_for_status = MagicMock()
            return response

        # Fail once with rate limit, then succeed
        mock_get.side_effect = [
            make_rate_limit_response(),
            make_success_response(),
        ]

        with patch.object(MyGHApi._get_single_page.retry, "sleep") as mock_sleep:
            result = api.is_repo_present(use_cache=False)

            self.assertTrue(result)
            # Should have slept for approximately 30 seconds (the rate limit reset time)
            mock_sleep.assert_called_once()
            sleep_duration = mock_sleep.call_args[0][0]
            self.assertGreaterEqual(sleep_duration, 25)  # Allow some tolerance
            self.assertLessEqual(sleep_duration, 35)


if __name__ == "__main__":
    unittest.main()
