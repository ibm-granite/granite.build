'use client'

import { Breadcrumb, BreadcrumbItem } from '@carbon/react'
import Link from 'next/link'

export interface Crumb {
  label: string
  to?: string
}

interface Props {
  crumbs: Crumb[]
}

/** Renders a Carbon Breadcrumb. Last crumb is always marked as current page. */
export function PageHeader({ crumbs }: Props) {
  return (
    <Breadcrumb noTrailingSlash style={{ marginBottom: '1rem' }}>
      {crumbs.map((c, i) => {
        const isCurrent = i === crumbs.length - 1
        return (
          <BreadcrumbItem key={i} isCurrentPage={isCurrent}>
            {c.to && !isCurrent ? <Link href={c.to}>{c.label}</Link> : c.label}
          </BreadcrumbItem>
        )
      })}
    </Breadcrumb>
  )
}
