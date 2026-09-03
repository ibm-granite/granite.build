'use client'

import { PageHeader } from '@granite-build/ui-core/components/PageHeader'
import { AutotunexTabs } from '@granite-build/ui-core/components/AutotunexTabs'
import { ConfigurationsTable } from '@granite-build/ui-core/components/ConfigurationsTable'
import { DatasetsTable } from '@granite-build/ui-core/components/DatasetsTable'
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
