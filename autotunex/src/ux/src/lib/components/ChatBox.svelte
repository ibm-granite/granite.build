<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import '@carbon/ai-chat/dist/es/web-components/cds-aichat-container/index.js';
	import { API } from '$lib/api';
	import {
		MessageResponseTypes,
		BusEventType,
		type CustomSendMessageOptions,
		type MessageResponse,
		type MessageRequest,
		type ChatInstance,
		type PublicConfig,
		type HistoryItem,
		type TextItem,
		type PartialItemChunk,
		type CompleteItemChunk,
		type FinalResponseChunk,
		CornersType
	} from '@carbon/ai-chat';
	import { forceUpdate } from '$lib/store';
	import { appState, updateJob } from '$lib/app';

	const api = new API();

	let chat = {
		history: [] as HistoryItem[],
		context: {} as Record<string, unknown>
	};

	// Stable id for this chat session — sent with every request so the backend
	// can keep per-thread LangGraph state (including ToolMessages) across turns.
	// Rotated when the user hits "restart conversation" in the chat header.
	let threadId = crypto.randomUUID();

	let chatInstance: ChatInstance | null = null;

	function doWelcomeText(instance: ChatInstance, request: MessageRequest) {
		const WELCOME_TEXT =
			'Hello! I can tune models or provide information about tuning experiments. How can I help you?';
		const welcome_output = {
			output: {
				generic: [
					{
						id: crypto.randomUUID(),
						request_id: request.id,
						response_type: MessageResponseTypes.TEXT,
						text: WELCOME_TEXT
					} as TextItem
				]
			}
		};
		record(welcome_output);
		instance.messaging.addMessage(welcome_output);
	}

	const record = (message: MessageRequest | MessageResponse) => {
		const createRecord = {
			message: { ...message },
			time: new Date().toISOString()
		};
		chat.history.push(createRecord);
	};

	function buildHistoryMessages(currentText: string) {
		const history_messages = chat.history
			.filter(
				(h: HistoryItem) =>
					(h as any).message?.input?.text || (h as any).message?.output?.generic?.[0]?.text
			)
			.map((h: HistoryItem) => {
				if ((h as any).message?.input?.text)
					return { role: 'user', content: (h as any).message.input.text };
				return {
					role: 'assistant',
					content: (h as any).message.output.generic[0].text
				};
			});
		history_messages.push({ role: 'user', content: currentText });
		return history_messages;
	}

	async function customSendMessage(
		request: MessageRequest,
		requestOptions: CustomSendMessageOptions,
		instance: ChatInstance
	) {
		instance.updateInputIsDisabled(true);
		record(request);

		if (request.input.message_type === 'event') {
			instance.updateInputIsDisabled(false);
			return;
		}
		if (!request.input.text) {
			doWelcomeText(instance, request);
			instance.updateInputIsDisabled(false);
			return;
		}

		const response_id = crypto.randomUUID();
		const final_id = crypto.randomUUID();

		// Per-tool status chips use fresh ids so text doesn't concatenate across tools.
		let currentStatusId: string | null = null;
		let closedStatusIds = new Set<string>();
		let answerText = '';

		const closeStatus = async (statusId: string | null) => {
			if (!statusId || closedStatusIds.has(statusId)) return;
			closedStatusIds.add(statusId);
			const completeChunk: CompleteItemChunk = {
				complete_item: {
					response_type: MessageResponseTypes.TEXT,
					text: '',
					streaming_metadata: { id: statusId }
				} as unknown as CompleteItemChunk['complete_item'],
				streaming_metadata: { response_id }
			};
			await instance.messaging.addMessageChunk(completeChunk);
		};

		// Forward Carbon's timeout abort into our fetch controller.
		const stream = api.startChatStream(
			buildHistoryMessages(request.input.text),
			chat.context,
			{
				onToolStart: async (_name, label) => {
					// Close any prior status chip so the next one renders as a fresh line.
					if (currentStatusId) await closeStatus(currentStatusId);
					const statusId = crypto.randomUUID();
					currentStatusId = statusId;
					const chunk: PartialItemChunk = {
						partial_item: {
							response_type: MessageResponseTypes.TEXT,
							text: `🔧 ${label}`,
							streaming_metadata: { id: statusId }
						} as unknown as PartialItemChunk['partial_item'],
						streaming_metadata: { response_id }
					};
					await instance.messaging.addMessageChunk(chunk);
				},
				onToolEnd: () => {
					/* status chip is closed when the next tool starts or on done */
				},
				onToken: async (text) => {
					answerText += text;
					const chunk: PartialItemChunk = {
						partial_item: {
							response_type: MessageResponseTypes.TEXT,
							text,
							streaming_metadata: { id: final_id }
						} as unknown as PartialItemChunk['partial_item'],
						streaming_metadata: { response_id }
					};
					await instance.messaging.addMessageChunk(chunk);
				},
				onContext: (ctx) => {
					chat.context = ctx;
				},
				onRefresh: async (target) => {
					// Push new/updated jobs into the shared store so the Tunings table
					// reflects them immediately whether the view is currently mounted or not.
					if (target === 'tunings') {
						try {
							const jobs = await api.getJobs();
							for (const job of jobs) updateJob(job);
							// Invalidate the "already loaded" flag so a future re-entry refetches cleanly.
							appState.update((prev) => ({ ...prev, isTuningsLoaded: true }));
						} catch (e) {
							console.error('Failed to refresh tunings after chat-triggered start:', e);
						}
					}
					// Fallback nudge for any view gated by {#key $forceUpdate}.
					forceUpdate.update((prev) => prev + 1);
				},
				onDone: async () => {
					if (currentStatusId) await closeStatus(currentStatusId);

					const bot_response: MessageResponse = {
						id: response_id,
						request_id: request.id,
						output: {
							generic: [
								{
									response_type: MessageResponseTypes.TEXT,
									text: answerText,
									streaming_metadata: { id: final_id },
									message_options: {
										feedback: {
											is_on: true,
											id: request.id,
											show_positive_details: false,
											show_negative_details: true,
											show_prompt: true
										}
									}
								} as TextItem
							]
						},
						context: chat.context
					};

					const finalChunk: FinalResponseChunk = { final_response: bot_response };
					await instance.messaging.addMessageChunk(finalChunk);
					record(bot_response);
				},
				onError: async (message) => {
					if (currentStatusId) await closeStatus(currentStatusId);
					const errorResponse: MessageResponse = {
						id: response_id,
						request_id: request.id,
						output: {
							generic: [
								{
									response_type: MessageResponseTypes.TEXT,
									text: `⚠️ ${message || 'Sorry, I hit an error. Please try again.'}`
								} as TextItem
							]
						}
					};
					const finalChunk: FinalResponseChunk = { final_response: errorResponse };
					await instance.messaging.addMessageChunk(finalChunk);
					record(errorResponse);
				}
			},
			threadId
		);

		const onAbort = () => stream.abort();
		if (requestOptions?.signal) {
			if (requestOptions.signal.aborted) stream.abort();
			else requestOptions.signal.addEventListener('abort', onAbort, { once: true });
		}

		try {
			await stream.done;
		} finally {
			requestOptions?.signal?.removeEventListener?.('abort', onAbort);
			instance.updateInputIsDisabled(false);
		}
	}

	async function customLoadHistory(_instance: ChatInstance) {
		if (localStorage.getItem('chat_history')) {
			let chat_history = localStorage.getItem('chat_history');
			chat_history = JSON.parse(chat_history as string);
			return chat_history as unknown as HistoryItem[];
		}
		return [];
	}

	function onBeforeRender(instance: ChatInstance) {
		chatInstance = instance;
		// Start a fresh server-side thread whenever the user restarts the chat
		// so the new conversation doesn't inherit stale tool results.
		instance.on?.({
			type: BusEventType.RESTART_CONVERSATION,
			handler: () => {
				threadId = crypto.randomUUID();
				chat.history = [];
				chat.context = {};
			}
		});
	}

	const config: PublicConfig = {
		messaging: {
			// Streaming begins almost immediately, so lower the built-in indicator window.
			messageLoadingIndicatorTimeoutSecs: 2,
			messageTimeoutSecs: 120,
			customSendMessage,
			customLoadHistory
		},
		layout: {
			corners: CornersType.ROUND
		},
		header: {
			showRestartButton: true,
			title: 'AutoTuneX Assistant'
		},
		assistantName: 'AutoTuneX Assistant',
		openChatByDefault: false,
		onError: (data) => {
			// Surface catastrophic errors as a panel so the user isn't left staring at a silent widget.
			try {
				chatInstance?.updateCatastrophicErrorPanel?.({
					isOpen: true,
					title: 'Assistant unavailable',
					bodyText: (data as any)?.errorType
						? `The chat hit an error (\`${
								(data as any).errorType
						  }\`). Please refresh and try again.`
						: 'The chat hit an unexpected error. Please refresh and try again.'
				});
			} catch (e) {
				console.error('Failed to render catastrophic error panel', e);
			}
		}
	};
</script>

<cds-aichat-container {config} {onBeforeRender}></cds-aichat-container>
