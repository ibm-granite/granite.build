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

/**
 * Reconcile a `selectedIds` shadow with the rows a Carbon `DataTable` is
 * currently showing.
 *
 * Carbon's selection is uncontrolled and it prunes itself: `normalize()`
 * (`@carbon/react/lib/components/DataTable/tools/normalize.js`) copies
 * `isSelected` forward for every row id it already knows, and drops the entry
 * entirely for ids that are no longer in `rows`. `selectedRows` is then derived
 * from the *current* `rowIds`, so a row that scrolls out of the page/search
 * result leaves Carbon's selection on its own, while a row that survives stays
 * ticked and nothing outside the table can untick it.
 *
 * So a shadow must be pruned, not emptied. Emptying it left rows visibly ticked
 * with an empty shadow: the batch bar still read "N items selected" while Delete
 * confirmed a count of 0, deleted nothing and closed as a success — and in the
 * settings tables it also emptied the list the `anyUndeletable` guard is computed
 * from, re-enabling Delete for a system configuration or an in-use dataset.
 *
 * Returns `prev` unchanged when nothing was dropped, so this is safe to call
 * from an effect on every render.
 */
export function pruneSelection(prev: string[], visibleIds: Iterable<string>): string[] {
  const visible = visibleIds instanceof Set ? visibleIds : new Set(visibleIds)
  const next = prev.filter((id) => visible.has(id))
  return next.length === prev.length ? prev : next
}
