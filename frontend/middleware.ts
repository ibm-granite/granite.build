import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Auth gating is handled client-side in ClientShell (auth lives in localStorage,
// which is inaccessible to middleware). This middleware only exists to ensure
// /login and /auth/* are always reachable without any server-side interference.
export function middleware(_request: NextRequest) {
  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
