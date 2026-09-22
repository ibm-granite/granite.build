'use client'

import { useQuery } from '@tanstack/react-query'
import { getJobByBuildId } from '@granite-build/ui-core/api/autotunex'
import { listSpaces } from '@granite-build/ui-core/api/gbserver'
import type { JobDetail } from '@granite-build/ui-core/types'

/**
 * The AutoTuneX tuning job linked to this build, or null when there isn't one.
 *
 * The build page used to answer this from build tags, but the tag comes from
 * AutoTuneX's `gb_tags` setting — renameable, extendable to a comma-separated
 * list, and empty disables tagging entirely — so the tag was a guess at a value
 * this code cannot see, and changing that setting silently removed every AutoTuneX
 * panel. Ask the authoritative endpoint instead, for every build.
 *
 * EVERY failure means "no linked job", and renders nothing. `getJobByBuildId` maps
 * 404 to null; this hook additionally swallows the 502 gbserver's proxy returns
 * when AutoTuneX is not deployed, the 401/403 a viewer without access gets, and
 * plain network errors. A build page in a deployment without AutoTuneX has to look
 * exactly as it did before AutoTuneX existed — an error banner on every build would
 * be worse than an absent tab — so loading and failure are deliberately
 * indistinguishable and neither `isLoading` nor `isError` is returned.
 */
export function useLinkedTuningJob(buildId: string): {
  job: JobDetail | null
  scope: 'own' | 'all'
} {
  // Same "admin of at least one space" gate the tunings and settings tables use —
  // admins get scope=all so a job linked to another user's build still resolves.
  const { data: spaces = [], isPending: spacesPending } = useQuery({
    queryKey: ['spaces'],
    queryFn: listSpaces,
  })
  const isAdmin = spaces.some((s) => s.is_admin)
  const scope: 'own' | 'all' = isAdmin ? 'all' : 'own'

  const { data } = useQuery({
    queryKey: ['autotunex-job-by-build', buildId, isAdmin],
    queryFn: () => getJobByBuildId(buildId, scope),
    // `isAdmin` is part of the key and starts false while `spaces` is in flight.
    // Firing before it settles ran this lookup twice for an admin — once at
    // scope=own, then again at scope=all under a new key. Wait for spaces.
    // (A failed `listSpaces` also settles, leaving scope=own, so this cannot hang.)
    enabled: Boolean(buildId) && !spacesPending,
    // Nothing is rendered on failure, so a retry buys no visible recovery — and on
    // a deployment without AutoTuneX every build page would pay one extra request
    // for a 502 that cannot succeed. That's one, not react-query's library default
    // of three, because this app's own QueryClient (ClientShell) already sets
    // `retry: 1`. react-query's default refetchOnWindowFocus still heals a
    // transient blip.
    retry: false,
    // TrialsTable polls trials based on job.status, so a frozen status here means
    // that poll never stops. Moved unchanged from AutoTuneXJobPanels.
    refetchInterval: (query) => {
      const s = query.state.data?.status
      return s && new Set(['running', 'pending']).has(s) ? 15_000 : false
    },
  })

  return { job: data ?? null, scope }
}
