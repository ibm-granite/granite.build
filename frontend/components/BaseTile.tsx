'use client'

import { Button, Tile, InlineLoading } from "@carbon/react";
import { Restart } from "@carbon/icons-react";

interface BaseTileProps {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  style?: React.CSSProperties;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export function BaseTile({ title, action, children, style, onRefresh, isRefreshing }: BaseTileProps) {
  return (
    <Tile style={{ padding: 0, ...style }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "1rem 1rem 0",
        }}
      >
        <h5 style={{ margin: 0, flex: "1 1 auto", minWidth: 0 }}>{title}</h5>
        {(action || onRefresh) && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.25rem", flexShrink: 0, marginLeft: "0.5rem" }}>
            {action}
            {onRefresh && (
              isRefreshing ? (
                <div style={{ width: '2rem', height: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <InlineLoading />
                </div>
              ) : (
                <Button
                  kind="ghost"
                  size="sm"
                  hasIconOnly
                  renderIcon={Restart}
                  iconDescription="Refresh"
                  tooltipPosition="bottom"
                  onClick={onRefresh}
                />
              )
            )}
          </div>
        )}
      </div>
      <div style={{ padding: "1rem", overflow: "hidden" }}>{children}</div>
    </Tile>
  );
}
