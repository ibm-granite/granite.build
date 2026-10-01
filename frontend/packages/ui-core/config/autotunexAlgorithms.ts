import type { AlgorithmDetail, AlgorithmOption, TuningGoal } from '../types'

// ── Step 0 goal questionnaire ─────────────────────────────────────────────────

export const GOAL_OPTIONS: {
  id: TuningGoal
  title: string
  sub_title: string
  description: string
  dataDescription: string
}[] = [
  {
    id: 'sft',
    title: 'Supervised fine-tuning',
    sub_title: 'Teach the model a specific task or style',
    description:
      'Train your model to follow instructions, generate specific outputs, or adopt a writing style. Best for tasks with clear input/output pairs.',
    dataDescription: 'Input/output pairs (e.g., instruction and response)',
  },
  {
    id: 'offline_rl',
    title: 'Preference learning',
    sub_title: 'Align the model with human feedback',
    description:
      'Train your model to prefer better responses over worse ones using preference data. Useful for improving response quality based on human judgment.',
    dataDescription: 'Prompts with preferred and rejected responses',
  },
  {
    id: 'online_rl',
    title: 'Reinforcement learning',
    sub_title: 'Let the model learn from reward signals',
    description:
      'The model generates its own responses and improves based on automated scoring. Great when you want the model to explore and find better answers on its own.',
    dataDescription: 'Prompts only (a reward model scores the responses)',
  },
]

// ── Algorithm catalogue (Step 0 detail cards) ─────────────────────────────────

export const ALGORITHM_DETAILS: AlgorithmDetail[] = [
  {
    id: 'lora',
    name: 'LoRA',
    category: 'sft',
    recommended: true,
    shortDescription:
      'Low-Rank Adaptation. Efficient fine-tuning by training small adapter weights. Best balance of quality and efficiency.',
    requiredColumns: ['input', 'output'],
  },
  {
    id: 'sft',
    name: 'SFT',
    category: 'sft',
    recommended: false,
    shortDescription:
      'Full Supervised Fine-Tuning. Updates all model weights for maximum quality, but requires more compute and memory.',
    requiredColumns: ['input', 'output'],
  },
  {
    id: 'alora',
    name: 'aLoRA',
    category: 'sft',
    recommended: false,
    shortDescription:
      'Adaptive LoRA. Dynamically adjusts rank allocation during training for optimal parameter efficiency.',
    requiredColumns: ['input', 'output'],
  },
  {
    id: 'lokr',
    name: 'LoKR',
    category: 'sft',
    recommended: false,
    shortDescription:
      'Low-Rank adaptation with Kronecker product. Memory-efficient alternative to LoRA using decomposed weight matrices.',
    requiredColumns: ['input', 'output'],
  },
  {
    id: 'loha',
    name: 'LoHA',
    category: 'sft',
    recommended: false,
    shortDescription:
      'Low-Rank adaptation with Hadamard product. Combines element-wise and low-rank factorization for expressive adapters.',
    requiredColumns: ['input', 'output'],
  },
  {
    id: 'vera',
    name: 'VeRA',
    category: 'sft',
    recommended: false,
    shortDescription:
      'Vector-based Random matrix Adaptation. Uses shared frozen random matrices with small trainable scaling vectors.',
    requiredColumns: ['input', 'output'],
  },
  {
    id: 'dpo',
    name: 'DPO',
    category: 'offline_rl',
    recommended: true,
    shortDescription:
      'Direct Preference Optimization. Simple, stable training from preference pairs without a separate reward model.',
    requiredColumns: ['prompt', 'chosen', 'rejected'],
  },
  {
    id: 'kto',
    name: 'KTO',
    category: 'offline_rl',
    recommended: false,
    shortDescription:
      'Kahneman-Tversky Optimization. Works with binary feedback (good/bad) instead of paired preferences.',
    requiredColumns: ['prompt', 'completion', 'label'],
  },
  {
    id: 'grpo',
    name: 'GRPO',
    category: 'online_rl',
    recommended: true,
    shortDescription:
      'Group Relative Policy Optimization. Learns from relative reward rankings of multiple responses. No critic model needed.',
    requiredColumns: ['prompt'],
  },
  {
    id: 'ppo',
    name: 'PPO',
    category: 'online_rl',
    recommended: false,
    shortDescription:
      'Proximal Policy Optimization. Classic RL approach with a value function. Proven but requires more memory.',
    requiredColumns: ['prompt'],
  },
  {
    id: 'dapo',
    name: 'DAPO',
    category: 'online_rl',
    recommended: false,
    shortDescription:
      'Decoupled Advantage Policy Optimization. Improves on PPO with decoupled clipping for more stable training.',
    requiredColumns: ['prompt'],
  },
]

// Grouped fallback used when an algorithm id isn't found in ALGORITHM_DETAILS
export const ALGORITHM_OPTIONS: AlgorithmOption[] = [
  { id: 'lora', name: 'SFT (LoRA / aLoRA / LoKR / ...)', category: 'sft', requiredColumns: ['input', 'output'] },
  { id: 'dpo', name: 'DPO', category: 'offline_rl', requiredColumns: ['prompt', 'chosen', 'rejected'] },
  { id: 'kto', name: 'KTO', category: 'offline_rl', requiredColumns: ['prompt', 'completion', 'label'] },
  { id: 'ppo', name: 'PPO', category: 'online_rl', requiredColumns: ['prompt'] },
  { id: 'grpo', name: 'GRPO', category: 'online_rl', requiredColumns: ['prompt'] },
  { id: 'dapo', name: 'DAPO', category: 'online_rl', requiredColumns: ['prompt'] },
]

// Maps an algorithm id to the dataset-type key the backend's
// /autotune_dataset_types response groups columns under.
export const ALGORITHM_TO_DATASET_TYPE: Record<string, string> = {
  sft: 'dataset_type_a',
  lora: 'dataset_type_a',
  alora: 'dataset_type_a',
  lokr: 'dataset_type_a',
  loha: 'dataset_type_a',
  vera: 'dataset_type_a',
  dpo: 'dataset_type_b',
  kto: 'dataset_type_c',
  ppo: 'dataset_type_d',
  grpo: 'dataset_type_d',
  dapo: 'dataset_type_d',
}

// Example dataset rows for each format, shown in the Step 0 summary panel
export const DATASET_EXAMPLES: Record<string, Record<string, string>[]> = {
  sft: [
    {
      input: 'Summarize this article about climate change...',
      output: 'The article discusses the impact of rising temperatures...',
    },
    { input: 'Translate to French: Hello, how are you?', output: 'Bonjour, comment allez-vous?' },
  ],
  dpo: [
    {
      prompt: 'Explain quantum computing',
      chosen: 'Quantum computing leverages quantum mechanical phenomena...',
      rejected: 'I am not sure about that topic.',
    },
    {
      prompt: 'Write a haiku about spring',
      chosen: 'Cherry blossoms fall / Gentle rain on morning grass / New life awakens',
      rejected: 'Spring is nice and warm.',
    },
  ],
  kto: [
    {
      prompt: 'Explain quantum computing',
      completion: 'Quantum computing leverages quantum mechanical phenomena...',
      label: 'true',
    },
    { prompt: 'What is 2+2?', completion: 'The answer is probably 5.', label: 'false' },
  ],
  online_rl: [
    { prompt: 'Write a Python function that sorts a list using merge sort' },
    { prompt: 'Explain the difference between TCP and UDP protocols' },
  ],
}
