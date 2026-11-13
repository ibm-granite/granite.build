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

import { PageHeader } from '@granite-build/ui-core/components/PageHeader'
import { AutotunexTabs } from '@granite-build/ui-core/components/autotunex/shared/AutotunexTabs'
import { ConfigurationsTable } from '@granite-build/ui-core/components/autotunex/settings/ConfigurationsTable'
import { DatasetsTable } from '@granite-build/ui-core/components/autotunex/settings/DatasetsTable'
import styles from './Settings.module.scss'

export default function SettingsPage() {
  return (
    <div style={{ padding: '1.5rem' }}>
      <PageHeader crumbs={[{ label: 'Model Customization', to: '/dashboard/autotunex' }, { label: 'Settings' }]} />
      <AutotunexTabs active="settings" />

      <div className={styles.grid}>
        <section className={styles.section}>
          <ConfigurationsTable />
        </section>

        <section className={styles.section}>
          <DatasetsTable />
        </section>
      </div>
    </div>
  )
}
