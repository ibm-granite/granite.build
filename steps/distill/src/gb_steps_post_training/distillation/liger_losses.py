import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_distillation import LigerFusedLinearDistillationBase


class LigerFusedLinearSkewedJSDFunction(LigerFusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(student_logits, teacher_logits, beta=0.5, alpha=0.0, use_kl_interpolation=False):
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if alpha > 0:
            if beta == 0:
                skewed_student = torch.logaddexp(
                    student_log_probs + math.log(1 - alpha),
                    teacher_log_probs + math.log(alpha),
                )
                return F.kl_div(skewed_student, teacher_log_probs, reduction="sum", log_target=True)
            elif beta == 1:
                skewed_teacher = torch.logaddexp(
                    student_log_probs + math.log(alpha),
                    teacher_log_probs + math.log(1 - alpha),
                )
                return (torch.exp(student_log_probs) * (student_log_probs - skewed_teacher)).sum()
            else:
                skewed_student = torch.logaddexp(
                    student_log_probs + math.log(1 - alpha),
                    teacher_log_probs + math.log(alpha),
                )
                skewed_teacher = torch.logaddexp(
                    student_log_probs + math.log(alpha),
                    teacher_log_probs + math.log(1 - alpha),
                )
                fwd = F.kl_div(skewed_student, teacher_log_probs, reduction="sum", log_target=True)
                rev = (torch.exp(student_log_probs) * (student_log_probs - skewed_teacher)).sum()
                return (1 - beta) * fwd + beta * rev
        else:
            if beta == 0:
                return F.kl_div(student_log_probs, teacher_log_probs, reduction="sum", log_target=True)
            elif beta == 1:
                return (torch.exp(student_log_probs) * (student_log_probs - teacher_log_probs)).sum()
            elif use_kl_interpolation:
                fkl = F.kl_div(student_log_probs, teacher_log_probs, reduction="sum", log_target=True)
                rkl = (torch.exp(student_log_probs) * (student_log_probs - teacher_log_probs)).sum()
                return (1 - beta) * fkl + beta * rkl
            else:
                log_mean = torch.logsumexp(
                    torch.stack([student_log_probs + math.log(1 - beta), teacher_log_probs + math.log(beta)], dim=0),
                    dim=0,
                )
                student_kl = F.kl_div(log_mean, student_log_probs, reduction="sum", log_target=True)
                teacher_kl = F.kl_div(log_mean, teacher_log_probs, reduction="sum", log_target=True)
                return beta * teacher_kl + (1 - beta) * student_kl

    @classmethod
    def forward(
        cls,
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor,
        teacher_bias: torch.Tensor,
        weight_hard_loss: float = 0.0,
        weight_soft_loss: float = 1.0,
        beta: float = 0.5,
        alpha: float = 0.0,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
        return_soft_hard_loss: bool = False,
        use_kl_interpolation: bool = False,
    ):
        return super().forward(
            cls=cls,
            ctx=ctx,
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            target=true_labels,
            student_bias=student_bias,
            teacher_bias=teacher_bias,
            chunk_size=chunk_size,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            beta=beta,
            alpha=alpha,
            ignore_index=ignore_index,
            temperature=temperature,
            compiled=compiled,
            return_soft_hard_loss=return_soft_hard_loss,
            use_kl_interpolation=use_kl_interpolation,
        )

    @staticmethod
    def backward(ctx, grad_output, *args):
        grads = LigerFusedLinearDistillationBase.backward(ctx, grad_output, *args)[:6]
        return (
            *grads,
            None,  # teacher_bias
            None,  # weight_hard_loss
            None,  # weight_soft_loss
            None,  # beta
            None,  # alpha
            None,  # ignore_index
            None,  # temperature
            None,  # compiled
            None,  # chunk_size
            None,  # return_soft_hard_loss
            None,  # use_kl_interpolation
        )


class LigerFusedLinearSkewedJSDLoss(torch.nn.Module):
    def __init__(
        self,
        weight_hard_loss: float = 0.0,
        weight_soft_loss: float = 1.0,
        beta: float = 0.5,
        alpha: float = 0.0,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
        return_soft_hard_loss: bool = False,
        use_kl_interpolation: bool = False,
    ):
        super().__init__()
        assert temperature != 0, "Temperature cannot be 0."
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.compiled = compiled
        self.beta = beta
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.return_soft_hard_loss = return_soft_hard_loss
        self.use_kl_interpolation = use_kl_interpolation

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor = None,
        teacher_bias: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return LigerFusedLinearSkewedJSDFunction.apply(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            true_labels,
            student_bias,
            teacher_bias,
            self.weight_hard_loss,
            self.weight_soft_loss,
            self.beta,
            self.alpha,
            self.ignore_index,
            self.temperature,
            self.compiled,
            self.chunk_size,
            self.return_soft_hard_loss,
            self.use_kl_interpolation,
        )
