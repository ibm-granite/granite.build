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

import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
	const env = loadEnv(mode, process.cwd(), '');

	// Dev-only proxy targets. `/local` keeps a localhost default; the deployment
	// targets are read from env so no specific hosts are baked into source.
	// Proxies with an empty target are dropped (Vite rejects `target: ''`).
	const rawProxies: Record<string, { target: string | undefined; prefix: string }> = {
		'/local': { target: env.PROXY_LOCAL_TARGET || 'http://localhost:8000', prefix: '/local' },
		'/stage': { target: env.PROXY_STAGE_TARGET, prefix: '/stage' },
		'/prod': { target: env.PROXY_PROD_TARGET, prefix: '/prod' }
	};

	const proxyConfig = Object.fromEntries(
		Object.entries(rawProxies)
			.filter(([, cfg]) => !!cfg.target)
			.map(([key, cfg]) => [
				key,
				{
					target: cfg.target,
					changeOrigin: true,
					secure: false,
					rewrite: (path: string) => path.replace(new RegExp(`^${cfg.prefix}`), '')
				}
			])
	);

	return {
		plugins: [sveltekit()],
		preview: {
			port: 3400,
			host: true
			// proxy: proxyConfig
		},
		server: {
			proxy: proxyConfig
		}
	};
});
