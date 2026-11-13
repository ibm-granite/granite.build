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
