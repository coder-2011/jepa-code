import torch
import torch.nn.functional as F
from torch import nn


class LLMJEPAModel(nn.Module):
    def __init__(self, backbone, lambda_jepa=1.0, gamma_lm=1.0, jepa_metric="cosine"):
        super().__init__()

        if jepa_metric not in {"cosine", "mse", "l2"}:
            raise ValueError("jepa_metric must be one of: cosine, mse, l2")

        self.backbone = backbone
        self.lambda_jepa = float(lambda_jepa)
        self.gamma_lm = float(gamma_lm)
        self.jepa_metric = jepa_metric

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
        )
        source_outputs = self.backbone(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        )
        target_outputs = self.backbone(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            output_hidden_states=True,
        )

        source_embeddings = self._last_token_embeddings(
            source_outputs.hidden_states[-1],
            source_attention_mask,
            source_last_index,
        )
        target_embeddings = self._last_token_embeddings(
            target_outputs.hidden_states[-1],
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
