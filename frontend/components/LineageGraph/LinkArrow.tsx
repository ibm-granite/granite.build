'use client'

import { Edge } from '@carbon/charts-react'
import { path as d3Path } from 'd3-path'
import type { ElkExtendedEdge } from 'elkjs'

export default function LinkArrow(props: {
  link: ElkExtendedEdge
  color?: string
  markerStart?: string
  markerEnd?: string
  className?: string
}) {
  const { link, color, markerStart, markerEnd } = props
  const sections = link.sections![0]
  const path = d3Path()

  path.moveTo(sections.startPoint.x, sections.startPoint.y)

  if (sections.bendPoints) {
    sections.bendPoints.forEach((b) => {
      path.lineTo(b.x, b.y)
    })
  }

  path.lineTo(sections.endPoint.x, sections.endPoint.y)

  return (
    <Edge
      path={path.toString()}
      markerStart={markerStart}
      markerEnd={markerEnd}
      color={color || '#E0E0E0'}
      fill="none"
      className={props.className}
    />
  )
}
