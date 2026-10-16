import React from 'react'
import {
  Archive,
  BuildImage,
  DataSet,
  DocumentMultiple_02,
  ModelAlt,
  ObjectStorage,
  ParentChild,
  ServiceId,
  TransformBinary,
} from '@carbon/icons-react'

/**
 * Apache Arrow logo — three ChevronRight shapes (>>>) derived from Carbon's
 * ChevronRight 16px path (M11 8 6 13 5.3 12.3 9.6 8 5.3 3.7 6 3z), scaled to
 * 0.82× and spaced 0.5px apart inside a 16×16 Carbon icon grid.
 */
function ApacheArrowIcon({ color, size = 16 }: { color: string; size?: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      width={size}
      height={size}
      fill={color}
      aria-hidden="true"
      focusable="false"
    >
      <path d="M5.3 8 1.1 15 0.5 14 4.1 8 0.5 2 1.1 1z" />
      <path d="M10.4 8 6.2 15 5.6 14 9.2 8 5.6 2 6.2 1z" />
      <path d="M15.5 8 11.3 15 10.7 14 14.3 8 10.7 2 11.3 1z" />
    </svg>
  )
}

export interface ArtifactTypeConfig {
  icon: React.ReactElement
  color: string
  text: string
  url: string
}

export const ARTIFACT_TYPE_CONFIG: Record<string, ArtifactTypeConfig> = {
  // ── Gbserver artifact types ───────────────────────────────────────────────
  Table:    { icon: <ParentChild color="#8A3FFC" size={16} />,           color: '#8A3FFC', text: 'Table',    url: '' },
  Dataset:  { icon: <DataSet color="#198038" size={16} />,               color: '#198038', text: 'Dataset',  url: '' },
  Model:    { icon: <ModelAlt color="#0043CE" size={16} />,              color: '#0043CE', text: 'Model',    url: '/models/detail' },
  File:     { icon: <Archive color="#1192E8" size={16} />,               color: '#1192E8', text: 'File',     url: '' },
  Fileset:  { icon: <DocumentMultiple_02 color="#08bdba" size={16} />,   color: '#08bdba', text: 'Fileset',  url: '' },
  Build:    { icon: <BuildImage color="#0043CE" size={16} />,            color: '#0043CE', text: 'Build',    url: '' },
  Artifact: { icon: <ServiceId color="#0043CE" size={16} />,             color: '#0043CE', text: 'Artifact', url: '' },
  Bucket:   { icon: <ObjectStorage color="#D02670" size={16} />,         color: '#D02670', text: 'Bucket',   url: '' },

  // ── Data processing pipeline stage types ─────────────────────────────────
  parquet:     { icon: <DataSet color="#198038" size={16} />,            color: '#198038', text: 'Parquet',     url: '' },
  arrow:       { icon: <ApacheArrowIcon color="#1192E8" size={16} />,    color: '#1192E8', text: 'Arrow',       url: '' },
  megatron:    { icon: <ModelAlt color="#0043CE" size={16} />,           color: '#0043CE', text: 'Megatron',    url: '' },
  merged_text: { icon: <DocumentMultiple_02 color="#08bdba" size={16} />, color: '#08bdba', text: 'Merged Text', url: '' },
  merged_bin:  { icon: <TransformBinary color="#8A3FFC" size={16} />,            color: '#8A3FFC', text: 'Merged Bin',  url: '' },
}

/** Normalise API-style uppercase keys (e.g. "MODEL") to title case (e.g. "Model"). */
export function artifactTypeKey(rawType: string): string {
  return rawType.charAt(0).toUpperCase() + rawType.slice(1).toLowerCase()
}
