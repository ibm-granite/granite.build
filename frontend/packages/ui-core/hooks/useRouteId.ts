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

'use client'

import { useEffect, useState } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * A percent-encoded route segment, decoded; a stray `%` (not a valid escape)
 * yields null instead of throwing.
 */
export function decodeRouteSegment(s: string): string | null {
  try {
    return decodeURIComponent(s)
  } catch {
    return null
  }
}

/**
 * The entity id for a statically exported `<base>/_/` detail route.
 *
 * The real id lives in the ?id= query param, not location.hash — a hash read
 * in a mount-only effect breaks when navigating from one entity's page to
 * another, since both are the same "_" route to Next's router and a one-time
 * hash read never sees the new id.
 * useSearchParams() is reactive, but it isn't enough by itself: Next's router
 * patches window.history, so our own cosmetic replaceState below (stripping
 * the query param once we've adopted the id) makes useSearchParams() briefly
 * report no id again. Latching the id into state — only ever overwritten by
 * a new *non-empty* param value — survives that revert.
 *
 * Back/forward lands on the pretty URL the replaceState left, with no ?id=
 * and a fresh mount, so the id is read from the path there. The path segment
 * is written encoded and decoded on read; a malformed escape yields no id.
 */
export function useRouteId(base: string): string {
  const searchParams = useSearchParams()
  const pathname = usePathname()
  const pathId = pathname.match(new RegExp(`^${escapeRegExp(base)}/([^/]+)/?$`))?.[1]
  const paramId = searchParams.get('id') ?? (pathId && pathId !== '_' ? decodeRouteSegment(pathId) : null)
  const [id, setId] = useState(paramId ?? '')

  useEffect(() => {
    if (paramId && paramId !== id) {
      setId(paramId)
    }
  }, [paramId, id])

  useEffect(() => {
    if (id) {
      window.history.replaceState(null, '', `${base}/${encodeURIComponent(id)}/`)
    }
  }, [base, id])

  return id
}
