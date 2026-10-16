import { PageHeader } from '@granite-build/ui-core/components/PageHeader'
import { StartTuningWizard } from './StartTuningWizard'

export default function StartTuningPage() {
  return (
    <div style={{ padding: '1.5rem' }}>
      <PageHeader
        crumbs={[
          { label: 'Model Customization', to: '/dashboard/autotunex' },
          { label: 'Start Tuning' },
        ]}
      />
      <StartTuningWizard />
    </div>
  )
}
