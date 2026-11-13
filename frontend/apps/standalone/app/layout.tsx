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

import type { Metadata } from 'next'
import './globals.scss'
import { ClientShell } from '@granite-build/ui-core/components/ClientShell'

export const metadata: Metadata = {
  title: 'Granite.build',
}

// Runs before hydration so the dark-theme attribute is on <html> from the very
// first paint. SSR can never know the stored preference (no localStorage on the
// server), so without this, useTheme() would have to set the attribute itself
// during a component's render — a React-invisible DOM mutation that causes a
// hydration mismatch (React error #418) on every fresh load with dark mode
// stored, which React's mismatch recovery then reverts.
// Runs before hydration so a dark-mode viewer never sees a white flash. This
// duplicates parseStoredPreference/resolveTheme from lib/themePreference.ts
// because an inline <head> script cannot import a module — the two must be kept
// in step. An absent stored value means "follow the system".
const THEME_INIT_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem('gb-ui-theme');
    var pref = (stored === 'g10' || stored === 'g100' || stored === 'system') ? stored : 'system';
    var dark = pref === 'g100' || (pref === 'system'
      && typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-color-scheme: dark)').matches);
    if (dark) document.documentElement.setAttribute('data-carbon-theme', 'g100');
    document.documentElement.setAttribute('data-theme-pref', pref);
  } catch (e) {}
})();
`.trim()

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body>
        <ClientShell>{children}</ClientShell>
      </body>
    </html>
  )
}
