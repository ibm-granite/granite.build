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

import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import { InlineNotification, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getArtifactModelCard } from '@granite-build/ui-core/api/gbserver'

export function ModelCardPanel({ artifactId }: { artifactId: string }) {
  const { data: content, isLoading, error } = useQuery({
    queryKey: ['artifact-model-card', artifactId],
    queryFn: () => getArtifactModelCard(artifactId),
  })

  if (error) return <InlineNotification kind="error" title="Failed to load model card" subtitle={String(error)} style={{ margin: '1rem' }} />
  if (isLoading) return <div style={{ padding: '1.5rem' }}><SkeletonText paragraph lineCount={10} /></div>
  if (!content) return <p style={{ padding: '1.5rem', color: 'var(--cds-text-secondary, #525252)' }}>No model card available for this artifact.</p>

  return (
    <div style={{ padding: '1.5rem', maxWidth: '800px' }}>
      <ReactMarkdown remarkPlugins={[remarkBreaks]}>{content}</ReactMarkdown>
    </div>
  )
}
