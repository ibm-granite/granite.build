<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import { Tab, TabContent, Tabs } from 'carbon-components-svelte';
</script>

<Tabs>
	<Tab label="JSONL" />
	<Tab label="CSV" />
	<svelte:fragment slot="content">
		<TabContent>
			<p style="padding-bottom:0.5rem;">How to represented data in JSONL file.</p>
			<code>
				<pre style="padding: 8px;">{`{"input":<Input data>,"output":<Output data>}`}<br
					/>{`{"input":<Input data>,"output":<Output data>}`}<br
					/>{`{"input":<Input data>,"output":<Output data>}`}</pre>
			</code>
			<p style="padding-bottom:0.5rem; padding-top:1rem">Example of data in JSONL file.</p>
			<code>
				<pre
					style="padding: 8px;">{`{"input":"In 2007 Etteplan reported a turnover of EUR125 .2 m.","output":"neutral"}`}<br
					/>{`{"input":"BasWare is headquartered in Espoo , Finland .","output":"neutral"}`}<br
					/>{`{"input":"The parties have therefore agreed to leave Avena out of the deal .","output":"neutral"}`}<br
					/>{`{"input":"The turbines are expected to be launched by the end of 2012 .","output":"neutral"}`}</pre>
			</code>
		</TabContent>
		<TabContent>
			<p style="padding-bottom:0.5rem;">How to represented data in CSV file.</p>
			<code>
				<pre style="padding: 8px;">{`input, output`}<br />{`<Input_1>, <Output_1>`}<br
					/>{`<Input_2>, <Output_2>`}<br />{`<Input_3>, <Output_3>`}</pre>
			</code>
			<p style="padding-top:0.5rem; padding-bottom:0.5rem">Example of data in CSV file.</p>
			<code>
				<pre style="padding: 8px;">{`input, output`}<br
					/>{`In 2007 Etteplan reported a turnover of EUR125 .2 m., neutral`}<br
					/>{`The parties have therefore agreed to leave Avena out of the deal., neutral`}<br
					/>{`The turbines are expected to be launched by the end of 2012., neutral`}</pre>
			</code>
		</TabContent>
	</svelte:fragment>
</Tabs>
