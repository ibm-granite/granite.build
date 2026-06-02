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

"""Base notification adapter interface."""

import abc

from gbserver.types.buildevent import BuildEvent


class NotificationAdapter(abc.ABC):
    """Abstract base class for notification delivery adapters."""

    @abc.abstractmethod
    async def deliver(self, event: BuildEvent) -> bool:
        """Deliver a build event notification.

        Returns True if delivery succeeded, False otherwise.
        """
        ...
