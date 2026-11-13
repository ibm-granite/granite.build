# Copyright Granite.Build Authors
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

import atexit
import threading

from sqlalchemy import Engine, create_engine

from gbserver.types.constants import GBSERVER_SQL_ECHO
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class EngineCache(object):
    """
    Maintains a cache of engines created with URI and args.
    Also installs atexit handler to dispose all engines on exit.
    Generally a singleton instance of this should be used and is provided by get_singleton_engine_cache().
    """

    engines: dict[str, Engine]
    lock = threading.Lock()

    def __init__(self):
        self.engines = {}
        atexit.register(self.dispose_all)

    def get_engine(self, db_uri: str, **kwargs) -> Engine:
        """Get and/or create an engine for the given URI, args and kwargs provided to the initializer."""
        key = f"{db_uri}::{kwargs}"  # Note that this includes a pwd, so don't log it.
        with self.lock:
            engine = self.engines.get(key, None)
            if engine is None:
                if GBSERVER_SQL_ECHO:
                    logger.info("Enable SQL engine echo mode")
                engine = create_engine(
                    db_uri, **kwargs, echo=GBSERVER_SQL_ECHO, echo_pool=True
                )
                self.engines[key] = engine
            return engine

    def dispose_all(self):
        """Registered atexit handler to dispose all engines on exit."""
        with self.lock:
            for engine in self.engines.values():
                try:
                    logger.info(f"Begin disposing of engine: {engine}")
                    engine.dispose()
                    logger.info(f"Done disposing of engine: {engine}")
                except Exception as e:
                    logger.error(f"Error disposing engine: {engine} - {e}")
            self.engines = {}


__Singleton_EngineCache = EngineCache()


def get_singleton_engine_cache() -> EngineCache:
    return __Singleton_EngineCache
