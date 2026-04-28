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

"""
Tests for retry strategy configuration system.
"""

from typing import Self

import pytest

from gbserver.resilience import build_retry_strategies_from_config
from gbserver.resilience.strategies import (
    FileNotFoundRetryStrategy,
    NCCLErrorRetryStrategy,
    PodEvictionRetryStrategy,
    UnhealthyInsufficientPodsRetryStrategy,
)


class TestBuildRetryStrategiesFromConfig:
    """Tests for build_retry_strategies_from_config function."""

    def test_no_config_returns_all_strategies(self: Self) -> None:
        """Test that None config returns all available strategies with defaults."""
        strategies = build_retry_strategies_from_config(config=None)

        assert len(strategies) == 4
        assert any(
            isinstance(s, UnhealthyInsufficientPodsRetryStrategy) for s in strategies
        )
        assert any(isinstance(s, PodEvictionRetryStrategy) for s in strategies)
        assert any(isinstance(s, NCCLErrorRetryStrategy) for s in strategies)
        assert any(isinstance(s, FileNotFoundRetryStrategy) for s in strategies)

        # Check default object types (only for strategies that support it)
        for strategy in strategies:
            if hasattr(strategy, "object_types"):
                assert strategy.object_types == ["AppWrapper"]

    def test_no_config_with_custom_object_types(self: Self) -> None:
        """Test that custom object types are applied to strategies that support it."""
        strategies = build_retry_strategies_from_config(
            config=None,
            object_types=["AppWrapper", "Job", "Deployment"],
        )

        assert len(strategies) == 4
        for strategy in strategies:
            # NCCLErrorRetryStrategy and FileNotFoundRetryStrategy don't use object_types
            if hasattr(strategy, "object_types"):
                assert strategy.object_types == ["AppWrapper", "Job", "Deployment"]

    def test_single_strategy_config(self: Self) -> None:
        """Test configuration with a single strategy."""
        config = [
            {
                "type": "UnhealthyInsufficientPods",
                "object_types": ["AppWrapper"],
            }
        ]

        strategies = build_retry_strategies_from_config(config=config)

        assert len(strategies) == 1
        assert isinstance(strategies[0], UnhealthyInsufficientPodsRetryStrategy)
        assert strategies[0].object_types == ["AppWrapper"]

    def test_multiple_strategies_config(self: Self) -> None:
        """Test configuration with multiple strategies."""
        config = [
            {
                "type": "UnhealthyInsufficientPods",
                "object_types": ["AppWrapper", "Job"],
            },
            {
                "type": "PodEviction",
                "object_types": ["AppWrapper"],
                "avoid_eviction_nodes": True,
            },
        ]

        strategies = build_retry_strategies_from_config(config=config)

        assert len(strategies) == 2

        # Check UnhealthyInsufficientPods strategy
        unhealthy_strategy = next(
            s
            for s in strategies
            if isinstance(s, UnhealthyInsufficientPodsRetryStrategy)
        )
        assert unhealthy_strategy.object_types == ["AppWrapper", "Job"]

        # Check PodEviction strategy
        eviction_strategy = next(
            s for s in strategies if isinstance(s, PodEvictionRetryStrategy)
        )
        assert eviction_strategy.object_types == ["AppWrapper"]
        assert eviction_strategy.avoid_eviction_nodes is True

    def test_pod_eviction_with_node_avoidance_disabled(self: Self) -> None:
        """Test PodEviction strategy with node avoidance explicitly disabled."""
        config = [
            {
                "type": "PodEviction",
                "object_types": ["AppWrapper"],
                "avoid_eviction_nodes": False,
            }
        ]

        strategies = build_retry_strategies_from_config(config=config)

        assert len(strategies) == 1
        assert isinstance(strategies[0], PodEvictionRetryStrategy)
        assert strategies[0].avoid_eviction_nodes is False

    def test_missing_type_field(self: Self) -> None:
        """Test that strategies without 'type' field are skipped."""
        config = [
            {
                "object_types": ["AppWrapper"],
                # Missing 'type' field
            },
            {
                "type": "PodEviction",
                "object_types": ["Job"],
            },
        ]

        strategies = build_retry_strategies_from_config(config=config)

        # Should only have the PodEviction strategy
        assert len(strategies) == 1
        assert isinstance(strategies[0], PodEvictionRetryStrategy)

    def test_unknown_strategy_type(self: Self) -> None:
        """Test that unknown strategy types are skipped."""
        config = [
            {
                "type": "UnknownStrategy",
                "object_types": ["AppWrapper"],
            },
            {
                "type": "PodEviction",
                "object_types": ["Job"],
            },
        ]

        strategies = build_retry_strategies_from_config(config=config)

        # Should only have the PodEviction strategy
        assert len(strategies) == 1
        assert isinstance(strategies[0], PodEvictionRetryStrategy)

    def test_default_object_types_used_when_not_specified(self: Self) -> None:
        """Test that default object types are used if not specified in strategy config."""
        config = [
            {
                "type": "UnhealthyInsufficientPods",
                # No object_types specified
            }
        ]

        strategies = build_retry_strategies_from_config(
            config=config,
            object_types=["Job", "Deployment"],
        )

        assert len(strategies) == 1
        assert strategies[0].object_types == ["Job", "Deployment"]

    def test_empty_config_returns_default_strategy(self: Self) -> None:
        """Test that empty config list returns default strategy."""
        config = []

        strategies = build_retry_strategies_from_config(config=config)

        # Should return at least the default strategy
        assert len(strategies) >= 1
        assert any(
            isinstance(s, UnhealthyInsufficientPodsRetryStrategy) for s in strategies
        )

    def test_only_pod_eviction_enabled(self: Self) -> None:
        """Test configuration with only PodEviction strategy enabled."""
        config = [
            {
                "type": "PodEviction",
                "object_types": ["AppWrapper", "Job"],
                "avoid_eviction_nodes": False,
            }
        ]

        strategies = build_retry_strategies_from_config(config=config)

        assert len(strategies) == 1
        assert isinstance(strategies[0], PodEvictionRetryStrategy)
        assert strategies[0].object_types == ["AppWrapper", "Job"]
        assert strategies[0].avoid_eviction_nodes is False

    def test_realistic_production_config(self: Self) -> None:
        """Test a realistic production configuration."""
        config = [
            {
                "type": "UnhealthyInsufficientPods",
                "object_types": ["AppWrapper", "Job"],
            },
            {
                "type": "PodEviction",
                "object_types": ["AppWrapper", "Job"],
                "avoid_eviction_nodes": False,  # Cluster-wide resource issue
            },
        ]

        strategies = build_retry_strategies_from_config(
            config=config,
            object_types=[
                "AppWrapper"
            ],  # This is ignored since each strategy specifies its own
        )

        assert len(strategies) == 2

        # Verify both strategies are configured correctly
        unhealthy_strategy = next(
            s
            for s in strategies
            if isinstance(s, UnhealthyInsufficientPodsRetryStrategy)
        )
        assert unhealthy_strategy.object_types == ["AppWrapper", "Job"]

        eviction_strategy = next(
            s for s in strategies if isinstance(s, PodEvictionRetryStrategy)
        )
        assert eviction_strategy.object_types == ["AppWrapper", "Job"]
        assert eviction_strategy.avoid_eviction_nodes is False
