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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type


class LineageService(ABC):
    @abstractmethod
    def emit_event(self, event: Dict) -> None:
        pass

    @abstractmethod
    def search_lineage_by_tags(
        self, tags: List[str], limit: int = 10, offset: int = 0
    ) -> Tuple[int, List[Dict]]:
        pass

    @abstractmethod
    def get_artifact_graph(
        self,
        artifact_name: Optional[str] = None,
        artifact_url: Optional[str] = None,
        artifact_type: Optional[str] = None,
        max_depth: int = 10,
        direction: str = "downstream",
    ) -> Optional[Dict]:
        pass


class LineageServiceFactory:
    _registry: Dict[str, Type[LineageService]] = {}

    @staticmethod
    def create(service_type: str) -> LineageService:
        if not LineageServiceFactory._registry:
            from gbserver.lineage.wandb_service import WandBLineageService

            LineageServiceFactory._registry["wandb"] = WandBLineageService
        if service_type not in LineageServiceFactory._registry:
            raise ValueError(f"Unsupported lineage provider: {service_type}")
        return LineageServiceFactory._registry[service_type]()
