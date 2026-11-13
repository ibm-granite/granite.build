// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import type { GbTask } from '../../types'

/**
 * The `hf://` URI of a job's tuning output, or null when the output is not on the
 * Hub (a local `file://` run, or no artifact registered yet).
 *
 * Read from the tuning task's `artifact_uri` because that is the location the
 * Results file list comes from (AutoTuneX `services/assets.py`
 * `_resolve_source`), so a link built from it describes the same repo as the
 * table.
 */
export function tuningHfArtifactUri(tasks: GbTask[] | undefined): string | null {
  const uri = tasks?.find((task) => task.task_type === 'TUNING')?.artifact_uri
  return uri?.startsWith('hf://') ? uri : null
}
