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
