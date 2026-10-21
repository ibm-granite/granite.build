# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from trl.experimental.gold.gold_config import GOLDConfig


@dataclass
class CustomGOLDConfig(GOLDConfig):
    r"""
    Custom configuration class that extends [`GOLDConfig`] with additional parameters.
    """

    use_ce_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use standard cross entropy loss instead of distillation loss (ToDo: teacher is still loaded, remove this for efficiency)."},
    )
    use_distillm2: bool = field(
        default=False,
        metadata={"help": "Whether to use DILSTILLM-2 loss (comparative) computation."},
    )
    use_distillm2_like: bool = field(
        default=False,
        metadata={"help": "Whether to use DILSTILLM-2 like loss (non-comparative) computation."},
    )
    use_reversed_distillm2_like: bool = field(
        default=False,
        metadata={"help": "Whether to use reversed DILSTILLM-2 like loss (non-comparative) computation."},
    )
    use_kl_interpolation: bool = field(
        default=False,
        metadata={"help": "When True and 0 < beta < 1 with alpha == 0 (and no adaptive KLD / distillm2-like), "
                  "use the convex FKL/RKL interpolation `(1 - beta) * FKL + beta * RKL` instead of the "
                  "standard mixture-distribution generalized JSD. This makes the beta sweep directly "
                  "comparable to the alpha > 0 regime, which already uses the same interpolation form."},
    )
    alpha: float = field(
        default=0.0,
        metadata={"help": "Alpha parameter for skewing KL and RKL."},
    )
    alpha_schedule: str = field(
        default="constant",
        metadata={"help": "Schedule for alpha: 'constant' or 'linear'. When 'linear', alpha transitions from alpha_init to alpha."},
    )
    alpha_init: float = field(
        default=None,
        metadata={"help": "Initial value of alpha when using linear schedule. If None, uses the alpha parameter."},
    )
    clip_alpha: float = field(
        default=0.0,
        metadata={"help": "Reward clipping parameter for sampled OPD loss. When > 0, clips reward at log(clip_alpha) / (1 - clip_alpha). Separate from alpha (skewing)."},
    )
    use_adaptive_kld: bool = field(
        default=False,
        metadata={"help": "Whether to use adakptive KL divergence (https://arxiv.org/abs/2404.02657) in the loss computation."},
    )
    lmbda_schedule: str = field(
        default="constant",
        metadata={"help": "Schedule for lmbda: 'constant' or 'linear'. When 'linear', lmbda increases from lmbda_init to lmbda_end."},
    )
    lmbda_init: float = field(
        default=None,
        metadata={"help": "Initial value of lmbda when using linear schedule. If None, uses the lmbda parameter."},
    )
    beta_schedule: str = field(
        default="constant",
        metadata={"help": "Schedule for beta: 'constant' or 'linear'. When 'linear', beta increases from beta_init to beta."},
    )
    beta_init: float = field(
        default=None,
        metadata={"help": "Initial value of beta when using linear schedule. If None, uses the beta parameter."},
    )
    instruction_template: str = field(
        default="<|start_of_role|>user<|end_of_role|>",
        metadata={"help": "Role marker for user message."},
    )
    response_template: str = field(
        default="<|start_of_role|>assistant<|end_of_role|>",
        metadata={"help": "Role marker for assistant message."},
    )
    last_message_only: bool = field(
        default=False,
        metadata={"help": "Whether to train only on the last message during off-policy training."},
    )
    use_hidden_loss: bool = field(
        default=False,
        metadata={"help": "Whether to add auxiliary hidden state matching loss (MSE after projection)."},
    )
    hidden_loss_gamma: float = field(
        default=0.05,
        metadata={"help": "Weight for the hidden state matching loss. Final weight is gamma * (1 - current_lmbda)."},
    )
    hidden_loss_layers: Optional[list[int]] = field(
        default=None,
        metadata={"help": "List of layer indices to match (0-indexed decoder layers). If None, uses last layer only."},
    )
    use_sampled_opd_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use sampled on-policy distillation loss (REINFORCE-style policy gradient on sampled tokens)."},
    )
    opd_switch_steps: float = field(
        default=0.0,
        metadata={"help": "Fraction of total training steps (0~1) after which to switch from alpha-clipped OPD to entropy-filtered OPD. 0.0 means never switch."},
    )
    opd_entropy_threshold: float = field(
        default=1.0,
        metadata={"help": "Fraction (0~1] of tokens to keep by student entropy (batch-global) after switching. E.g., 0.2 = top 20% highest-entropy tokens."},
    )
    opd_importance_sampling: bool = field(
        default=False,
        metadata={"help": "Enable truncated importance sampling for sampled_opd_loss to correct vLLM/HF distribution mismatch."},
    )
    opd_is_epsilon_low: float = field(
        default=0.5,
        metadata={"help": "Lower bound for IS weight truncation (tokens with weight < this are masked out)."},
    )
    opd_is_epsilon_high: float = field(
        default=2.0,
        metadata={"help": "Upper bound for IS weight truncation (tokens with weight > this are masked out)."},
    )
    uld_hybrid_sampled_opd_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for sampled OPD (REINFORCE-style) loss in hybrid ULD mode. "
                  "Computed at alignment group level. 0.0 disables it."},
    )
    use_old_uld_loss: bool = field(
        default=False,
        metadata={"help": "Use original ULDLoss instead of CustomULDLoss for debugging/comparison."},
    )
    student_eos_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Student EOS token ID. If None, auto-detected from tokenizer."},
    )
    teacher_eos_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Teacher EOS token ID. If None, auto-detected from tokenizer."},
    )
    use_liger_fused_jsd: bool = field(
        default=False,
        metadata={"help": "Use Liger fused linear JSD loss to reduce GPU memory. "
                  "Unlike use_liger_kernel, this only applies the fused JSD loss without "
                  "patching model internals (safe for GraniteMoEHybrid/Mamba models)."},
    )
    max_alignment_group_size: int = field(
        default=10,
        metadata={"help": "Maximum number of tokens per side in an alignment group. "
                  "Groups exceeding this are treated as alignment failures (e.g. Unicode "
                  "mismatch); the alignment is truncated at the first oversized group. "
                  "Set to 0 to disable."},
    )
    visualize_sampled_opd_reward: bool = field(
        default=False,
        metadata={"help": "Print HTML snippet that colors each aligned group by its sampled OPD reward. "
                  "Blue = high reward, white = zero, red = low."},
    )
    uld_reuse_student_input: bool = field(
        default=False,
        metadata={"help": "Reuse student input_ids for teacher in ULD loss (requires identical tokenizers, jaccard=1.0). For debugging."},
    )
    uld_rebuild_student_input: bool = field(
        default=False,
        metadata={"help": "Retokenize student input using student's tokenizer in ULD loss. "
                  "Prevents student from exploiting non-standard tokenization to hack reward. "
                  "Incompatible with opd_importance_sampling and uld_reuse_student_input."},
    )
    uld_rebuild_student_input_with_teacher_template: bool = field(
        default=False,
        metadata={"help": "Retokenize student input using student's tokenizer after applying teacher's template."},
    )
    uld_align_at_bytes: bool = field(
        default=False,
        metadata={"help": "Use UTF-8 byte-level alignment in ULD cross-tokenizer matching. "
                  "Fixes emoji/multi-byte character alignment failures. "
                  "Assumes both tokenizers use GPT-2 style byte-to-unicode BPE."},
    )
    uld_teacher_enable_thinking: bool = field(
        default=False,
        metadata={"help": "Render the ULD teacher side with enable_thinking=True when splitting "
                  "prompt/completion. Required for <think> reasoning-distillation data: with the "
                  "default (False), the teacher prompt gets an empty <think></think> while the full "
                  "render has a filled <think>...</think>, so full.startswith(prompt) is False and the "
                  "prompt/completion slice is corrupted (teacher labels misalign -> ULD loss 0). The "
                  "turn-suffix used to strip the trailing <|im_end|>\\n is still computed with "
                  "enable_thinking=False, which matches the completion tail either way."},
    )
    uld_per_turn_regions: bool = field(
        default=False,
        metadata={"help": "Render ONE teacher supervised region per assistant turn instead of one "
                  "per row. Required whenever last_message_only=False, because the student's "
                  "collator then labels every assistant turn (utils.py:478-499) while the teacher "
                  "renders msgs[:-1] + the last turn: the first alignment group cannot close, every "
                  "row yields no groups, and the ULD loss is a real tensor of exactly zero (job "
                  "1492353 trained 100 steps at loss 0.0, checkpointing normally throughout). The "
                  "alternative -- last_message_only=True -- works but supervises 53.4% of the tokens "
                  "the collator labels, and the 46.6% it drops are every assistant turn but the "
                  "last, i.e. the intermediate tool calls on an agentic corpus. Default False keeps "
                  "every prior arm byte-identical."},
    )
    uld_teacher_template_kwargs: list[str] = field(
        default_factory=list,
        metadata={"help": "Extra chat-template variables for the PER-TURN teacher render, as "
                  "KEY=VALUE strings (true/false parsed as bools, everything else kept as a "
                  "string). Needed because a template may render a HISTORICAL assistant turn "
                  "differently from a final one, and per-turn supervision reads every turn out "
                  "of one full render: granite's template declares "
                  "`truncate_history_thinking` (default TRUE), which replaces a historical "
                  "turn's reasoning with `<think></think>`, so that turn's own text is absent "
                  "from the render and cannot be supervised at all -- 110 of 3000 real rows, "
                  "confirmed directly, zero with truncate_history_thinking=false. "
                  "Applied ONLY on the per-turn path, so the single-region arms render exactly "
                  "as they did before this field existed; a template that does not declare the "
                  "variable ignores it."},
    )
    opd_normalize_by_group: bool = field(
        default=False,
        metadata={"help": "Normalize group log-probs by group size in sampled OPD loss. "
                  "Divides each group's summed log-prob by its number of positions, "
                  "preventing N-to-1 groups from having disproportionately large rewards."},
    )
    uld_opd_normalize_reward_with_baseline: bool = field(
        default=False,
        metadata={"help": "Normalize sampled OPD rewards across the batch by subtracting "
                  "the mean (baseline) and dividing by std. Improves RL stability."},
    )
    opd_ignore_multi_loss: bool = field(
        default=False,
        metadata={"help": "When True, compute sampled OPD loss only on 1-to-1 alignment groups, "
                  "ignoring multi-token groups (1-to-N, N-to-1, N-to-N)."},
    )
    opd_ignore_unbalanced_loss: bool = field(
        default=False,
        metadata={"help": "When True, compute sampled OPD loss only on balanced alignment groups "
                  "(where len(s_group) == len(t_group)), ignoring unbalanced groups."},
    )
    uld_ignore_multi_loss: bool = field(
        default=False,
        metadata={"help": "When True, compute matched (JSD) and unmatched (sorted L1) losses only on "
                  "1-to-1 alignment groups, ignoring multi-token groups (1-to-N, N-to-1, N-to-N)."},
    )
    uld_matched_top_k: int = field(
        default=0,
        metadata={"help": "When > 0, compute Matched (JSD) loss only on the teacher's top-k tokens "
                  "within the matched (overlapping) vocabulary. Applied per-position. "
                  "Requires uld_ignore_multi_loss=true. 0 disables (uses full overlapping vocabulary)."},
    )
    uld_region_chunk_groups: int = field(
        default=0,
        metadata={"help": "When > 0, compute the region loss over this many ALIGNMENT GROUPS at a "
                  "time, each chunk wrapped in torch.utils.checkpoint so its full-vocabulary "
                  "probability tensors are freed and recomputed in backward instead of being held "
                  "until the whole row's backward. The loss and gradients are unchanged (both "
                  "reductions are sums over positions divided once by a position count); peak "
                  "memory becomes O(chunk) instead of O(row tokens), paid for with one extra "
                  "forward per chunk. Required for a 250k-vocab teacher on long reasoning rows: "
                  "at ~4.2 MiB of probability tensors per supervised position, a p90 row of this "
                  "corpus needs ~41 GiB whole-region. 0 keeps the historical behaviour. The "
                  "chunked path supports uld_renormalize_probs=true with uld_align_at_bytes, "
                  "uld_matched_top_k=0, uld_ignore_multi_loss=false, no OPD weight and explicit "
                  "hybrid weights, and RAISES on anything else rather than falling back."},
    )
    uld_renormalize_probs: bool = field(
        default=False,
        metadata={"help": "When True, compute matched (JSD) loss on renormalized distributions over the "
                  "matched vocabulary subset only, instead of using probabilities from full-vocab softmax. "
                  "For 1-to-1 alignment groups, this skips the expensive full-vocab softmax and alignment "
                  "group merge, working directly with logits at matched indices."},
    )
    is_gptoss_teacher: bool = field(
        default=False,
        metadata={"help": "When True, assume the teacher is a gpt-oss model."},
    )
    add_empty_analysis: bool = field(
        default=False,
        metadata={"help": "When True, add empty analysis channel in gpt-oss prompt."},
    )
    vllm_num_servers: int = field(
        default=1,
        metadata={"help": "Number of independent vLLM servers to use for on-policy generation. "
                  "When > 1, prompts are split across servers in parallel for higher throughput. "
                  "Each server runs on a separate node with its own NCCL communicator."},
    )
    overlap_generation: bool = field(
        default=False,
        metadata={"help": "Overlap vLLM generation with training. Step N trains on step N-1's "
                  "generated data while step N's generation runs concurrently in a background thread. "
                  "Requires vLLM server mode. Incompatible with use_distillm2 and use_uld_loss."},
    )
    vllm_temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for vLLM student rollouts. Decoupled from `temperature` "
                  "(which is used as the softmax temperature in the distillation loss)."},
    )
    reinit_sinks_to: Optional[float] = field(
        default=None,
        metadata={"help": "If set, reinitialize granite_swa attention sinks "
                  "(model.layers.*.self_attn.sinks) to this value at trainer init. "
                  "Used to lift sinks out of sigmoid-saturation regions (e.g. -6) "
                  "into a regime with non-vanishing gradient. None = no reinit."},
    )

    def __post_init__(self):
        super().__post_init__()

        # Validate alpha is non-negative
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in range 0-1.")

        # Validate alpha schedule parameters
        if self.alpha_schedule == "linear":
            if self.alpha_init is None:
                raise ValueError("When alpha_schedule='linear', alpha_init must be specified.")
            if not (0.0 <= self.alpha_init <= 1.0):
                raise ValueError("alpha_init must be in range [0, 1].")
        elif self.alpha_schedule != "constant":
            raise ValueError(f"alpha_schedule must be 'constant' or 'linear', got '{self.alpha_schedule}'.")

        # Validate lmbda schedule parameters
        if self.lmbda_schedule == "linear":
            if self.lmbda_init is None:
                raise ValueError("When lmbda_schedule='linear', lmbda_init must be specified.")
            if not (0.0 <= self.lmbda_init <= 1.0):
                raise ValueError("lmbda_init must be in range [0, 1].")
            if not (0.0 <= self.lmbda <= 1.0):
                raise ValueError("lmbda must be in range [0, 1].")
        elif self.lmbda_schedule != "constant":
            raise ValueError(f"lmbda_schedule must be 'constant' or 'linear', got '{self.lmbda_schedule}'.")

        # Validate clip_alpha
        if not (0.0 <= self.clip_alpha < 1.0):
            raise ValueError("clip_alpha must be in range [0.0, 1.0).")

        # Validate hidden loss parameters
        if self.use_hidden_loss and self.hidden_loss_gamma < 0:
            raise ValueError("hidden_loss_gamma must be >= 0.")

        # Validate sampled OPD loss requirements
        if self.use_sampled_opd_loss:
            if self.lmbda != 1.0:
                raise ValueError("use_sampled_opd_loss requires lmbda=1.0 (fully on-policy).")
            if not self.last_message_only:
                raise ValueError("use_sampled_opd_loss requires last_message_only=true.")
            if self.use_liger_fused_jsd:
                raise ValueError("use_sampled_opd_loss is incompatible with use_liger_fused_jsd.")
            if not (0.0 <= self.opd_switch_steps < 1.0):
                raise ValueError("opd_switch_steps must be in [0, 1).")
            if not (0.0 < self.opd_entropy_threshold <= 1.0):
                raise ValueError("opd_entropy_threshold must be in (0, 1].")
            if self.opd_importance_sampling:
                if self.opd_is_epsilon_low < 0:
                    raise ValueError("opd_is_epsilon_low must be >= 0.")
                if self.opd_is_epsilon_high <= self.opd_is_epsilon_low:
                    raise ValueError("opd_is_epsilon_high must be > opd_is_epsilon_low.")

        # Validate beta schedule parameters
        if self.beta_schedule == "linear":
            if self.beta_init is None:
                raise ValueError("When beta_schedule='linear', beta_init must be specified.")
            if not (0.0 <= self.beta_init <= 1.0):
                raise ValueError("beta_init must be in range [0, 1].")
        elif self.beta_schedule != "constant":
            raise ValueError(f"beta_schedule must be 'constant' or 'linear', got '{self.beta_schedule}'.")

        # Validate uld_rebuild_student_input compatibility
        # if self.uld_rebuild_student_input and self.opd_importance_sampling:
        #     raise ValueError("uld_rebuild_student_input is incompatible with opd_importance_sampling.")
        if self.uld_rebuild_student_input and self.uld_reuse_student_input:
            raise ValueError("uld_rebuild_student_input is incompatible with uld_reuse_student_input.")
        if self.uld_rebuild_student_input_with_teacher_template and self.uld_reuse_student_input:
            raise ValueError("uld_rebuild_student_input_with_teacher_template is incompatible with uld_reuse_student_input.")


        # Validate uld_per_turn_regions. Every one of these is a combination in which the
        # per-turn regions would be built and then silently mis-consumed, which is worse than
        # the bug the flag fixes: that one at least showed up as a loss of exactly zero.
        if self.uld_per_turn_regions:
            if not self.use_uld_loss:
                raise ValueError(
                    "uld_per_turn_regions requires use_uld_loss=true: it changes how the ULD "
                    "teacher's supervised regions are rendered, and nothing else reads them. "
                    "A same-vocab arm needs no teacher render at all -- the student's own "
                    "multi-span labels are already used as-is.")
            if not self.uld_use_hybrid_loss:
                raise ValueError(
                    "uld_per_turn_regions requires uld_use_hybrid_loss=true. Only the hybrid "
                    "path's loss (custom_gold_trainer._compute_distillation_loss) iterates "
                    "regions; the base path delegates to trl's, which derives ONE region per "
                    "row from _get_start_and_size_answers (first supervised index + total "
                    "supervised count) and would slice a single span straddling the user and "
                    "tool turns in between.")
            if self.last_message_only:
                raise ValueError(
                    "uld_per_turn_regions is incompatible with last_message_only=true. The two "
                    "are opposite answers to the same question: last_message_only makes the "
                    "student's collator label ONE span (utils.py:470-473) while per-turn "
                    "rendering gives the teacher one per assistant turn, so every row would "
                    "report a span-count mismatch and be supervised on its first turn only. "
                    "Set last_message_only=false -- supervising every turn is the point.")

        if self.uld_teacher_template_kwargs and not self.uld_per_turn_regions:
            raise ValueError(
                "uld_teacher_template_kwargs requires uld_per_turn_regions=true: it is applied "
                "only on the per-turn render, so setting it without that flag would be a silent "
                "no-op. The single-region path is deliberately left untouched -- these variables "
                "change how HISTORICAL turns render, which only per-turn supervision reads.")
        for kv in self.uld_teacher_template_kwargs:
            if "=" not in kv:
                raise ValueError(
                    f"uld_teacher_template_kwargs entries must be KEY=VALUE, got {kv!r}")

        # Validate uld_matched_top_k requires uld_ignore_multi_loss
        if self.uld_matched_top_k > 0 and not self.uld_ignore_multi_loss:
            raise ValueError("uld_matched_top_k requires uld_ignore_multi_loss=true.")

        # Validate uld_region_chunk_groups. The trainer's own gate raises at step 0 too, but the
        # combinations are decidable HERE, before ~10 minutes of teacher weight loading, and a
        # config that asks for chunking must never quietly get the whole-region path instead.
        if self.uld_region_chunk_groups < 0:
            raise ValueError("uld_region_chunk_groups must be >= 0 (0 disables chunking).")
        if self.uld_region_chunk_groups > 0:
            bad = []
            if not self.uld_align_at_bytes:
                bad.append("needs uld_align_at_bytes=true (the chunks ARE alignment groups)")
            if not self.uld_renormalize_probs:
                bad.append("needs uld_renormalize_probs=true")
            if self.uld_matched_top_k > 0:
                bad.append("uld_matched_top_k>0 unsupported")
            if self.uld_ignore_multi_loss:
                bad.append("uld_ignore_multi_loss unsupported")
            if self.uld_hybrid_sampled_opd_weight > 0:
                bad.append("uld_hybrid_sampled_opd_weight>0 unsupported (OPD reads whole-region tensors)")
            if self.uld_hybrid_matched_weight is None:
                bad.append("adaptive weights unsupported (set uld_hybrid_matched_weight)")
            if bad:
                raise ValueError(
                    "uld_region_chunk_groups=%d is outside the chunked region-loss path: %s"
                    % (self.uld_region_chunk_groups, "; ".join(bad)))

        # Validate add_empty_analysis
        if self.add_empty_analysis and not self.is_gptoss_teacher:
            raise ValueError("add_empty_analysis is incompatible with non-gpt-oss teacher.")

        # Validate vllm_num_servers
        if self.vllm_num_servers < 1:
            raise ValueError("vllm_num_servers must be >= 1.")

        # Validate overlap_generation compatibility
        if self.overlap_generation:
            if self.use_distillm2:
                raise ValueError("overlap_generation is incompatible with use_distillm2.")
            if self.use_uld_loss:
                raise ValueError("overlap_generation is incompatible with use_uld_loss.")
            if self.use_vllm and self.vllm_mode != "server":
                raise ValueError(
                    "overlap_generation requires vllm_mode='server'. "
                    "Colocate mode does not support async generation overlap."
                )

