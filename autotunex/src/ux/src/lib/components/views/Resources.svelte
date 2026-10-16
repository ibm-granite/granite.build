<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		StructuredList,
		StructuredListHead,
		StructuredListRow,
		StructuredListCell,
		StructuredListBody
	} from 'carbon-components-svelte';

	let resources = [
		{
			type: 'Inproceedings',
			title:
				'Bandit Limited Discrepancy Search and Application to Machine Learning Pipeline Optimization',
			author: 'A. Kishimoto, et. al.',
			publisher:
				'Proceedings of the AAAI Conference on Artificial Intelligence, 36(9), 10228-10237',
			year: '2022'
		},
		{
			type: 'Inproceedings',
			title: 'Searching for machine learning pipelines using a context-free grammar',
			author: 'Marinescu, Radu, et al.',
			publisher: 'Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 10.',
			year: '2021'
		},
		{
			type: 'Workshop',
			title: 'Automated Decision Optimization with Reinforcement Learning',
			author: 'Marinescu, Radu, et al.',
			publisher: 'Workshop at AAAI Conference on Artificial Intelligence',
			year: '2022'
		},
		{
			type: 'Inproceedings',
			title: 'Automated AI For Decision Optimization with Reinforcement Learning',
			author: 'Subramaniam, Shankar, et al.',
			publisher: 'AAAI Conference on Artificial Intelligence',
			year: '2023'
		},
		{
			type: 'Inproceedings',
			title: 'AutoDOViz: Human-Centered Automation for Decision Optimization',
			author: 'Weidele, Daniel Karl I., et al.',
			publisher: 'Proceedings of the 28th International Conference on Intelligent User Interfaces',
			year: '2023'
		},
		{
			type: 'Inproceedings',
			title:
				'AutoAIViz: opening the blackbox of automated artificial intelligence with conditional parallel coordinates',
			author: 'Weidele, Daniel Karl I., et al.',
			publisher: 'Proceedings of the 25th International Conference on Intelligent User Interfaces',
			year: '2020'
		},
		{
			type: 'Journal',
			title: 'AutoRL X: Automated Reinforcement Learning on the Web',
			author: 'Franke, Loraine, et al. ',
			publisher: 'ACM Transactions on Interactive Intelligent Systems',
			year: '2023'
		},
		{
			type: 'Demo/Code',
			title: 'Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning',
			author:
				'C. Poth, H. Sterz, P. Indraneil, S. Purkayastha, L. Engländer, T. Imhof, I. Vulić, S. Ruder, I. Gurevych, J. Pfeiffer',
			publisher: 'https://aclanthology.org/2023.emnlp-demo.13',
			year: '2023'
		},
		{
			type: 'Code',
			title: 'PEFT: State-of-the-art Parameter-Efficient Fine-Tuning',
			author: 'S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, B. Bossan',
			publisher: 'https://github.com/huggingface/peft',
			year: '2022'
		}
	];
</script>

<StructuredList condensed flush>
	<StructuredListHead>
		<StructuredListRow head>
			<StructuredListCell head>Type</StructuredListCell>
			<StructuredListCell head>Year</StructuredListCell>
			<StructuredListCell head>Reference</StructuredListCell>
		</StructuredListRow>
	</StructuredListHead>
	<StructuredListBody style="overflow: scroll;">
		{#each resources as resource}
			<StructuredListRow>
				<StructuredListCell>{resource.type}</StructuredListCell>
				<StructuredListCell>
					{resource.year}
				</StructuredListCell>
				<StructuredListCell>
					{resource.author}. "{resource.title}".<br />
					{resource.publisher}
				</StructuredListCell>
			</StructuredListRow>
		{/each}
	</StructuredListBody>
</StructuredList>
