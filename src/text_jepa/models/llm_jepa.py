from copy import deepcopy
import random
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from ..attention_masks import (
    block_diagonal_causal_additive_mask,
    causal_additive_mask,
    stack_additive_attention_masks,
)
from ..objectives.stp import random_span_batch_loss
from ..utils.ema import update_ema


class LLMJEPAModel(nn.Module):
    def __init__(
        self,
        backbone,
        lambda_jepa=1.0,
        gamma_lm=1.0,
        jepa_metric="cosine",
        ema_momentum=0.996,
        objective_mode="paired",
        student_packing="separate",
        stp_samples=1,
        stp_max_span_length=None,
        stp_min_span_length=1,
        stp_length_adjustment=None,
        stp_layer=-1,
        stp_linear_predictor=False,
    ):
        super().__init__()

        if jepa_metric not in {"cosine", "mse", "l2"}:
            raise ValueError("jepa_metric must be one of: cosine, mse, l2")
        if objective_mode not in {"paired", "stp_random_span"}:
            raise ValueError("objective_mode must be one of: paired, stp_random_span")
        if student_packing not in {"separate", "additive-mask"}:
            raise ValueError("student_packing must be one of: separate, additive-mask")
        if stp_samples <= 0:
            raise ValueError("stp_samples must be positive")
        if stp_min_span_length <= 0:
            raise ValueError("stp_min_span_length must be positive")

        self.backbone = backbone
        self.lambda_jepa = float(lambda_jepa)
        self.gamma_lm = float(gamma_lm)
        self.jepa_metric = jepa_metric
        self.ema_momentum = float(ema_momentum)
        self.objective_mode = objective_mode
        self.student_packing = student_packing
        self.uses_target_backbone = self.objective_mode == "paired"
        self.stp_samples = int(stp_samples)
        self.stp_max_span_length = stp_max_span_length
        self.stp_min_span_length = int(stp_min_span_length)
        self.stp_length_adjustment = stp_length_adjustment
        self.stp_layer = int(stp_layer)
        self._stp_rng = random.Random(torch.initial_seed())

        hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
        if self.jepa_metric == "cosine" and hidden_size is not None and hidden_size <= 2:
            warnings.warn(
                "Cosine JEPA with hidden_size<=2 is geometrically degenerate: final hidden states can collapse "
                "to a single line after normalization, making cosine similarities saturate at +/-1.",
                stacklevel=2,
            )

        self.target_backbone = None
        if self.uses_target_backbone:
            self.target_backbone = deepcopy(backbone)
            self.target_backbone.requires_grad_(False)
            update_ema(self.target_backbone, self.backbone, momentum=0.0)

        self.stp_predictor = None
        if stp_linear_predictor:
            if hidden_size is None:
                raise ValueError("stp_linear_predictor requires a backbone with a known hidden size")
            self.stp_predictor = nn.Linear(hidden_size, hidden_size, bias=False)

    def get_extra_state(self):
        return {
            "stp_rng_state": self._stp_rng.getstate(),
        }

    def set_extra_state(self, state):
        if not state:
            return
        stp_rng_state = state.get("stp_rng_state")
        if stp_rng_state is not None:
            self._stp_rng.setstate(stp_rng_state)

    def _final_hidden_state(self, backbone, input_ids, attention_mask):
        base_model = getattr(backbone, "base_model", None)
        if base_model is not None:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is not None:
                return last_hidden_state

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("backbone must return either last_hidden_state or hidden_states")
        return hidden_states[-1]

    def _last_token_embeddings(self, hidden_states, attention_mask, indices=None):
        if indices is None:
            indices = attention_mask.long().sum(dim=1) - 1
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), indices]

    def _jepa_loss(self, source_embeddings, target_embeddings):
        if self.jepa_metric == "mse":
            return F.mse_loss(source_embeddings, target_embeddings)
        if self.jepa_metric == "l2":
            return torch.linalg.vector_norm(source_embeddings - target_embeddings, dim=-1).mean()
        cosine_similarity = F.cosine_similarity(source_embeddings, target_embeddings, dim=-1)
        return 1.0 - cosine_similarity.mean()

    def _build_additive_mask_stack(
        self,
        attention_mask,
        packed_source_length,
        packed_target_length,
        packed_attention_mask,
        *,
        packed_mode,
    ):
        sequence_length = attention_mask.shape[1]
        masks = []
        device = attention_mask.device
        for full_length in attention_mask.long().sum(dim=1).tolist():
            masks.append(
                causal_additive_mask(
                    sequence_length,
                    valid_length=int(full_length),
                    device=device,
                )
            )

        if packed_mode == "paired":
            for source_length, target_length in zip(packed_source_length.tolist(), packed_target_length.tolist()):
                masks.append(
                    block_diagonal_causal_additive_mask(
                        [int(source_length), int(target_length)],
                        sequence_length=sequence_length,
                        device=device,
                    )
                )
        elif packed_mode == "tube":
            for packed_length in packed_attention_mask.long().sum(dim=1).tolist():
                masks.append(
                    causal_additive_mask(
                        sequence_length,
                        valid_length=int(packed_length),
                        device=device,
                    )
                )
        else:
            raise ValueError(f"unsupported packed_mode: {packed_mode}")

        return stack_additive_attention_masks(masks, device=device)

    def _require_packed_views(self, packed_valid, *, mode_name):
        if packed_valid is None:
            raise ValueError(f"{mode_name} requires packed dataset fields")
        if not torch.all(packed_valid.to(torch.bool)):
            raise ValueError(
                f"{mode_name} requires source+target packed rows that fit within max_length; "
                "reduce predictors or max_length, or choose a mode that does not require packing"
            )

    def _run_student_additive(
        self,
        input_ids,
        attention_mask,
        labels,
        packed_input_ids,
        packed_attention_mask,
        packed_labels,
        packed_source_length,
        packed_target_length,
        *,
        packed_mode,
        hidden_state_index,
    ):
        additive_mask = self._build_additive_mask_stack(
            attention_mask,
            packed_source_length,
            packed_target_length,
            packed_attention_mask,
            packed_mode=packed_mode,
        )
        batch_size = input_ids.shape[0]
        outputs = self.backbone(
            input_ids=torch.cat([input_ids, packed_input_ids], dim=0),
            attention_mask=additive_mask,
            labels=torch.cat([labels, packed_labels], dim=0),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("student additive mode requires the backbone to return hidden_states")
        packed_hidden_states = hidden_states[hidden_state_index][batch_size:]
        return outputs, packed_hidden_states

    def _forward_paired(
        self,
        input_ids,
        attention_mask,
        labels,
        source_input_ids,
        source_attention_mask,
        target_input_ids,
        target_attention_mask,
        source_last_index=None,
        target_last_index=None,
        packed_input_ids=None,
        packed_attention_mask=None,
        packed_labels=None,
        packed_source_length=None,
        packed_target_length=None,
        packed_source_last_index=None,
        packed_valid=None,
    ):
        if self.student_packing == "additive-mask":
            self._require_packed_views(packed_valid, mode_name="paired additive-mask mode")
            lm_outputs, packed_hidden_states = self._run_student_additive(
                input_ids,
                attention_mask,
                labels,
                packed_input_ids,
                packed_attention_mask,
                packed_labels,
                packed_source_length,
                packed_target_length,
                packed_mode="paired",
                hidden_state_index=-1,
            )
            source_embeddings = self._last_token_embeddings(
                packed_hidden_states,
                packed_attention_mask,
                packed_source_last_index,
            )
        else:
            lm_outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            source_hidden_states = self._final_hidden_state(
                self.backbone,
                source_input_ids,
                source_attention_mask,
            )
            source_embeddings = self._last_token_embeddings(
                source_hidden_states,
                source_attention_mask,
                source_last_index,
            )

        with torch.no_grad():
            target_hidden_states = self._final_hidden_state(
                self.target_backbone,
                target_input_ids,
                target_attention_mask,
            )

        target_embeddings = self._last_token_embeddings(
            target_hidden_states,
            target_attention_mask,
            target_last_index,
        )

        lm_loss = lm_outputs.loss
        jepa_loss = self._jepa_loss(source_embeddings, target_embeddings)
        loss = self.gamma_lm * lm_loss + self.lambda_jepa * jepa_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "jepa_loss": jepa_loss,
            "source_embeddings": source_embeddings,
            "target_embeddings": target_embeddings,
        }

    def _forward_stp(
        self,
        input_ids,
        attention_mask,
        labels,
        packed_input_ids=None,
        packed_attention_mask=None,
        packed_labels=None,
        packed_source_length=None,
        packed_target_length=None,
        packed_source_span_start=None,
        packed_source_span_end=None,
        packed_target_span_start=None,
        packed_target_span_end=None,
        packed_valid=None,
        **_,
    ):
        self._require_packed_views(packed_valid, mode_name="STP mode")
        if self.student_packing == "additive-mask":
            lm_outputs, packed_hidden_states = self._run_student_additive(
                input_ids,
                attention_mask,
                labels,
                packed_input_ids,
                packed_attention_mask,
                packed_labels,
                packed_source_length,
                packed_target_length,
                packed_mode="tube",
                hidden_state_index=self.stp_layer,
            )
        else:
            lm_outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            packed_outputs = self.backbone(
                input_ids=packed_input_ids,
                attention_mask=packed_attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            packed_hidden_states = packed_outputs.hidden_states[self.stp_layer]

        stp_loss = random_span_batch_loss(
            packed_hidden_states,
            packed_source_length,
            packed_target_length,
            source_span_starts=packed_source_span_start,
            source_span_ends=packed_source_span_end,
            target_span_starts=packed_target_span_start,
            target_span_ends=packed_target_span_end,
            rng=self._stp_rng,
            samples=self.stp_samples,
            min_span_length=self.stp_min_span_length,
            max_span_length=self.stp_max_span_length,
            metric=self.jepa_metric,
            length_adjustment=self.stp_length_adjustment,
            predictor=self.stp_predictor,
        )
        lm_loss = lm_outputs.loss
        loss = self.gamma_lm * lm_loss + self.lambda_jepa * stp_loss
        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "jepa_loss": stp_loss,
            "stp_loss": stp_loss,
        }

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        source_input_ids,
        source_attention_mask,
        target_input_ids,
        target_attention_mask,
        source_last_index=None,
        target_last_index=None,
        packed_input_ids=None,
        packed_attention_mask=None,
        packed_labels=None,
        packed_source_length=None,
        packed_target_length=None,
        packed_source_last_index=None,
        packed_target_last_index=None,
        packed_source_span_start=None,
        packed_source_span_end=None,
        packed_target_span_start=None,
        packed_target_span_end=None,
        packed_valid=None,
    ):
        if self.objective_mode == "paired":
            return self._forward_paired(
                input_ids,
                attention_mask,
                labels,
                source_input_ids,
                source_attention_mask,
                target_input_ids,
                target_attention_mask,
                source_last_index=source_last_index,
                target_last_index=target_last_index,
                packed_input_ids=packed_input_ids,
                packed_attention_mask=packed_attention_mask,
                packed_labels=packed_labels,
                packed_source_length=packed_source_length,
                packed_target_length=packed_target_length,
                packed_source_last_index=packed_source_last_index,
                packed_valid=packed_valid,
            )
        return self._forward_stp(
            input_ids,
            attention_mask,
            labels,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
            source_last_index=source_last_index,
            target_last_index=target_last_index,
            packed_input_ids=packed_input_ids,
            packed_attention_mask=packed_attention_mask,
            packed_labels=packed_labels,
            packed_source_length=packed_source_length,
            packed_target_length=packed_target_length,
            packed_source_last_index=packed_source_last_index,
            packed_target_last_index=packed_target_last_index,
            packed_source_span_start=packed_source_span_start,
            packed_source_span_end=packed_source_span_end,
            packed_target_span_start=packed_target_span_start,
            packed_target_span_end=packed_target_span_end,
            packed_valid=packed_valid,
        )
