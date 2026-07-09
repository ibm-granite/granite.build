'use client'

import { CardNode, CardNodeColumn, CardNodeSubtitle, CardNodeTitle } from '@carbon/charts-react'
import {
  DataSet,
  DocumentMultiple_02,
  ModelAlt,
  ObjectStorage,
  Unknown,
} from '@carbon/icons-react'
import styles from './GraphNode.module.scss'
import type { ElkNodeEx } from './Graph'

interface NodeItemProps {
  node: ElkNodeEx
  onClick?: (node: ElkNodeEx) => void
  onMouseHover?: (node: ElkNodeEx | null) => void
  selectedNode?: ElkNodeEx
}

function getNodeConfig(type: string | undefined) {
  switch (type) {
    case 'Build':
      return { icon: null, color: undefined as string | undefined, bgColor: 'rgba(113,119,132,0.15)', subtitleLabel: 'Process' }
    case 'Model':
      return { icon: <ModelAlt color="#0043CE" size={16} />, color: '#0043CE', bgColor: 'rgba(0,67,206,0.15)', subtitleLabel: 'Model' }
    case 'Fileset':
      return { icon: <DocumentMultiple_02 color="#08bdba" size={16} />, color: '#08bdba', bgColor: 'rgba(8,189,186,0.15)', subtitleLabel: 'Fileset' }
    case 'Dataset':
      return { icon: <DataSet color="#198038" size={16} />, color: '#198038', bgColor: 'rgba(25,128,56,0.15)', subtitleLabel: 'Dataset' }
    case 'Bucket':
      return { icon: <ObjectStorage color="#D02670" size={16} />, color: '#D02670', bgColor: 'rgba(208,38,112,0.15)', subtitleLabel: 'Bucket' }
    default:
      return { icon: <Unknown color="#717784" size={16} />, color: '#717784' as string | undefined, bgColor: 'rgba(113,119,132,0.15)', subtitleLabel: undefined }
  }
}

function SkeletonNode({ direction }: { direction: 'left' | 'right' }) {
  return (
    <div className={direction === 'right' ? styles.skeletonNodeRight : styles.skeletonNodeLeft}>
      <CardNode color="transparent">
        <CardNodeColumn>
          <CardNodeTitle className={styles.skeletonTitle}>…</CardNodeTitle>
        </CardNodeColumn>
      </CardNode>
    </div>
  )
}

export default function GraphNode({ node, onClick, onMouseHover, selectedNode }: NodeItemProps) {
  const { x, y, height, width, type, title } = node

  if (type === 'skeleton-source') {
    return (
      <foreignObject transform={`translate(${x}, ${y})`} height={height} width={width}>
        <SkeletonNode direction="left" />
      </foreignObject>
    )
  }

  if (type === 'skeleton-target') {
    return (
      <foreignObject transform={`translate(${x}, ${y})`} height={height} width={width}>
        <SkeletonNode direction="right" />
      </foreignObject>
    )
  }

  const { icon, color, bgColor, subtitleLabel } = getNodeConfig(type)
  const isSelected = selectedNode?.id === node.id
  const isHighlighted = node.highlight

  const nodeWrapperClass = [
    styles.nodeWrapper,
    type === 'Build' ? styles.nodeWrapperBuildType : '',
    isSelected ? styles.nodeWrapperSelected : '',
    isHighlighted ? styles.nodeWrapperHighlighted : '',
    node.planned ? styles.nodeWrapperPlanned : '',
  ].filter(Boolean).join(' ')

  return (
    <foreignObject
      transform={`translate(${x}, ${y})`}
      height={height}
      width={width}
      className={styles.foreignObjectOverflow}
    >
      <div className={nodeWrapperClass} style={isSelected ? { '--node-selected-bg': bgColor } as React.CSSProperties : undefined}>
        <CardNode
          color={color}
          onClick={() => onClick && onClick(node)}
          onMouseEnter={() => onMouseHover && onMouseHover(node)}
          onMouseLeave={() => onMouseHover && onMouseHover(null)}
        >
          {icon && (
            <CardNodeColumn className={styles.iconColumn}>
              {icon}
            </CardNodeColumn>
          )}
          <CardNodeColumn className={styles.titleColumn}>
            <CardNodeTitle
              style={{
                fontSize: '14px',
                fontWeight: 600,
                lineHeight: '18px',
                color: node.planned ? 'var(--cds-text-secondary)' : 'var(--cds-text-primary)',
                fontStyle: node.planned ? 'italic' : undefined,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {title}
            </CardNodeTitle>
            {subtitleLabel && (
              <CardNodeSubtitle
                style={{ fontSize: '0.75rem', color: 'var(--cds-text-secondary)' }}
              >
                {subtitleLabel}
              </CardNodeSubtitle>
            )}
          </CardNodeColumn>
        </CardNode>
      </div>
    </foreignObject>
  )
}
