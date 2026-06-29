import axios from 'axios'
import type { Plan, LinkedBuild } from '@/types'

const client = axios.create({ baseURL: '/api/analytics/plans' })

client.interceptors.request.use((config) => {
  const auth = localStorage.getItem('gb-ui-auth')
  if (auth) {
    try {
      const { token } = JSON.parse(auth)
      if (token) config.headers['Authorization'] = `Bearer ${token}`
    } catch { /* ignore */ }
  }
  return config
})

export async function listPlans(): Promise<{ plans: Plan[]; total: number }> {
  const { data } = await client.get<{ plans: Plan[]; total: number }>('')
  return data
}

export async function getPlan(planId: string): Promise<{ plan: Plan; builds: LinkedBuild[] }> {
  const { data } = await client.get<{ plan: Plan; builds: LinkedBuild[] }>(`/${planId}`)
  return data
}
