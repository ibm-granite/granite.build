/**
 * Shared best-effort bulk delete for the AutoTuneX list views (tunings, data
 * sets, configurations), which all delete a multi-select one id at a time
 * because the API exposes no batch endpoint.
 */

/** A partial bulk delete: everything not in `failedIds` was removed server-side. */
export interface BulkDeleteError extends Error {
  failedIds: string[]
  /** The first underlying rejection, for status-code-specific messaging (e.g. 409). */
  firstError: unknown
}

export function isBulkDeleteError(err: unknown): err is BulkDeleteError {
  return err instanceof Error && Array.isArray((err as BulkDeleteError).failedIds)
}

/**
 * Deletes every id, collecting failures instead of aborting on the first.
 *
 * Aborting stranded the ids behind the failure: one in-use item mid-selection
 * meant the rest were silently never attempted, while the ones already deleted
 * stayed on screen. Attempting all of them keeps the outcome equal to what the
 * user asked for, and the thrown `failedIds` lets the caller narrow its
 * selection to the survivors so a retry never re-issues a DELETE for an id that
 * is already gone.
 */
export async function deleteEach(
  ids: string[],
  deleteOne: (id: string) => Promise<unknown>
): Promise<void> {
  const failedIds: string[] = []
  let firstError: unknown
  for (const id of ids) {
    try {
      await deleteOne(id)
    } catch (err) {
      failedIds.push(id)
      if (failedIds.length === 1) firstError = err
    }
  }
  if (failedIds.length === 0) return
  const err = new Error(
    `${failedIds.length} of ${ids.length} item(s) could not be deleted`
  ) as BulkDeleteError
  err.failedIds = failedIds
  err.firstError = firstError
  throw err
}
