from __future__ import annotations

import torch
from torch.nn import functional as F


def class_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    prototypes = []
    for class_id in classes:
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"No support samples for class {int(class_id)}")
        prototypes.append(embeddings[mask].mean(dim=0))
    return torch.stack(prototypes)


def cosine_logits(query: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return query @ prototypes.transpose(0, 1) / temperature


def remap_labels(labels: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    matches = labels[:, None] == classes[None, :]
    if not matches.any(dim=1).all():
        raise ValueError("Query contains a class not represented by the support set")
    return matches.float().argmax(dim=1)


def multilevel_episode(
    support_embeddings: list[torch.Tensor],
    support_labels: torch.Tensor,
    query_embeddings: list[torch.Tensor],
    query_labels: torch.Tensor | None,
    layer_weights: list[float],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not (len(support_embeddings) == len(query_embeddings) == len(layer_weights)):
        raise ValueError("Feature levels and layer_weights must have identical lengths")
    classes = torch.unique(support_labels, sorted=True)
    total_logits = None
    total_loss = None
    mapped = remap_labels(query_labels, classes) if query_labels is not None else None
    for support, query, weight in zip(support_embeddings, query_embeddings, layer_weights):
        prototypes = class_prototypes(support, support_labels, classes)
        logits = cosine_logits(query, prototypes, temperature)
        total_logits = logits * weight if total_logits is None else total_logits + logits * weight
        if mapped is not None:
            loss = F.cross_entropy(logits, mapped)
            total_loss = loss * weight if total_loss is None else total_loss + loss * weight
    assert total_logits is not None
    return total_logits, total_loss

