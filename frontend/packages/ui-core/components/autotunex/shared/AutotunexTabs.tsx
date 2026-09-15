'use client'

import Link from 'next/link'
import { Tabs, TabList, Tab } from '@carbon/react'

interface Props {
  active: 'tunings' | 'settings'
}

/**
 * Shared Tunings | Settings switcher rendered at the top of both the
 * AutoTuneX Tunings page and the Settings page. Each tab is a Next.js Link
 * so clicking navigates between the two real routes (the app uses static
 * export; there is no client-side view toggle). `active` is passed
 * explicitly by each page rather than derived from the pathname.
 */
export function AutotunexTabs({ active }: Props) {
  const selectedIndex = active === 'settings' ? 1 : 0
  return (
    <Tabs selectedIndex={selectedIndex}>
      <TabList aria-label="AutoTuneX sections">
        {/* @ts-expect-error Carbon Tab's as prop expects href to be passed through */}
        <Tab as={Link} href="/dashboard/autotunex">
          Tunings
        </Tab>
        {/* @ts-expect-error Carbon Tab's as prop expects href to be passed through */}
        <Tab as={Link} href="/dashboard/autotunex/settings">
          Settings
        </Tab>
      </TabList>
    </Tabs>
  )
}
