'use client'

import { useQuery } from '@tanstack/react-query'
import { getMe } from '../api/autotunex'

/**
 * Whether AutoTuneX treats the caller as an admin — the only thing that lets a
 * request use `scope=all`. Every AutoTuneX screen shares this one cached query.
 *
 * Any failure reads as "not an admin", which keeps every screen on `scope=own`:
 * an AutoTuneX older than /api/v1/auth/me 404s here, and a 401 or dropped
 * connection would fail the `scope=all` reads anyway. `retry: false` because
 * none of those heal on a retry, and callers that wait on `isPending` would sit
 * through it for nothing.
 */
export function useAutotunexIsAdmin(): { isAdmin: boolean; isPending: boolean } {
  const { data, isPending } = useQuery({
    queryKey: ['autotunex-me'],
    queryFn: getMe,
    retry: false,
  })
  return { isAdmin: data?.is_admin ?? false, isPending }
}
