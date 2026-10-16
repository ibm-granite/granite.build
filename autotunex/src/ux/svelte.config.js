import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/kit/vite';
import { optimizeImports } from 'carbon-preprocess-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: [vitePreprocess(), optimizeImports()],

	kit: {
		adapter: adapter({
			fallback: 'index.html'
		}),
		paths: {
			base: '/autotune'
		}
	}
};

export default config;
