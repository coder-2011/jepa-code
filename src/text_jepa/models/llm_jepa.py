from copy import deepcopy
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.ema import update_ema


class LLMJEPAModel(nn.Module):
    def __init__(
        self,
        backbone,
        lambda_jepa=1.0,
        gamma_lm=1.0,
        jepa_metric="cosine",
        ema_momentum=0.996,
    ):
        super().__init__()

        if jepa_metric not in {"cosine", "mse", "l2"}:
            raise ValueError("jepa_metric must be one of: cosine, mse, l2")

        self.backbone = backbone
        self.target_backbone = deepcopy(backbone)
        self.lambda_jepa = float(lambda_jepa)
        self.gamma_lm = float(gamma_lm)
        self.jepa_metric = jepa_metric
        self.ema_momentum = float(ema_momentum)

        hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
        if self.jepa_metric == "cosine" and hidden_size is not None and hidden_size <= 2:
            warnings.warn(
                "Cosine JEPA with hidden_size<=2 is geometrically degenerate: final hidden states can collapse "
                "to a single line after normalization, making cosine similarities saturate at +/-1.",
                stacklevel=2,
            )

        self.target_backbone.requires_grad_(False)
        update_ema(self.target_backbone, self.backbone, momentum=0.0)

    def _final_hidden_state(self, backbone, input_ids, attention_mask):
        base_model = getattr(backbone, "base_model", None)
        if base_model is None:
            raise ValueError("LLMJEPAModel requires a causal LM backbone with a `base_model` module")

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise ValueError("backbone.base_model(...) must return `last_hidden_state`")
        return last_hidden_state

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
    ):
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
        with torch.no_grad():
            target_hidden_states = self._final_hidden_state(
                self.target_backbone,
                target_input_ids,
                target_attention_mask,
            )

        source_embeddings = self._last_token_embeddings(
            source_hidden_states,
            source_attention_mask,
            source_last_index,
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
