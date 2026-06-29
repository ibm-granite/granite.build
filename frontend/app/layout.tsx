import type { Metadata } from 'next'
import './globals.scss'
import { ClientShell } from '@/components/ClientShell'

export const metadata: Metadata = {
  title: 'Granite.build',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ClientShell>{children}</ClientShell>
      </body>
    </html>
  )
}
