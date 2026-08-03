from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from .config import get, resolve_path
from .data import (
    CollaborativeDataset,
    EpisodeFactory,
    ImageCollection,
    add_generated_support,
    load_generated_pool,
)
from .models import build_model, load_encoder_weights
from .progress import tqdm
from .prototypical import multilevel_episode
from .utils import AverageMeter, load_checkpoint, mean_confidence_interval, save_checkpoint, write_json


def collection_from_config(config: dict[str, Any], domain: str) -> ImageCollection:
    section = get(config, f"data.{domain}", {})
    folder = section.get("folder")
    manifest = section.get("manifest")
    root = section.get("root")
    return ImageCollection.load(
        folder=resolve_path(config, folder) if folder else None,
        manifest=resolve_path(config, manifest) if manifest else None,
        root=resolve_path(config, root) if root else None,
        split=section.get("split"),
    )


def collaborative_train(config: dict[str, Any], device: torch.device) -> Path:
    source = collection_from_config(config, "source")
    image_size = int(get(config, "data.image_size", 224))
    aug = get(config, "augmentation", {})
    dataset = CollaborativeDataset(
        source,
        image_size=image_size,
        clip_limit=float(aug.get("clip_limit", 4.0)),
        tile_grid=int(aug.get("tile_grid", 8)),
        blur_kernel=int(aug.get("blur_kernel", 13)),
    )
    settings = get(config, "collaborative", {})
    loader = DataLoader(
        dataset,
        batch_size=int(settings.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(settings.get("workers", 4)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(settings.get("workers", 4)) > 0,
    )
    model = build_model(len(source.class_names), get(config, "model.architecture", "resnet10")).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-3)),
        momentum=float(settings.get("momentum", 0.9)),
        weight_decay=float(settings.get("weight_decay", 5e-4)),
    )
    criterion = torch.nn.CrossEntropyLoss()
    aug_weight = float(settings.get("augmented_loss_weight", 1.0))
    output = resolve_path(config, get(config, "output_dir", "outputs")) / "collaborative.pt"
    history = []
    for epoch in range(1, int(settings.get("epochs", 400)) + 1):
        model.train()
        loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
        progress = tqdm(loader, desc=f"Collaborative {epoch}", leave=False)
        for original, augmented, labels in progress:
            original = original.to(device, non_blocking=True)
            augmented = augmented.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_original = model(original)
            logits_augmented = model(augmented)
            loss = criterion(logits_original, labels) + aug_weight * criterion(logits_augmented, labels)
            loss.backward()
            optimizer.step()
            count = labels.numel()
            accuracy = (logits_original.argmax(1) == labels).float().mean().item()
            loss_meter.update(loss.item(), count)
            accuracy_meter.update(accuracy, count)
            progress.set_postfix(loss=f"{loss_meter.average:.4f}", acc=f"{accuracy_meter.average:.3f}")
        history.append({"epoch": epoch, "loss": loss_meter.average, "accuracy": accuracy_meter.average})
        save_checkpoint(
            output,
            stage="collaborative",
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            class_names=source.class_names,
            config=config,
            history=history,
        )
    return output


def meta_train(config: dict[str, Any], device: torch.device, checkpoint: str | Path | None = None) -> Path:
    source = collection_from_config(config, "source")
    settings = get(config, "meta_train", {})
    image_size = int(get(config, "data.image_size", 224))
    factory = EpisodeFactory(
        source,
        ways=int(settings.get("ways", 5)),
        shots=int(settings.get("shots", 5)),
        queries=int(settings.get("queries", 15)),
        image_size=image_size,
        train=True,
        chestx_style=True,
        seed=int(get(config, "seed", 1)) + 101,
    )
    model = build_model(len(source.class_names), get(config, "model.architecture", "resnet10")).to(device)
    checkpoint_path = Path(checkpoint) if checkpoint else resolve_path(
        config, get(config, "output_dir", "outputs")
    ) / "collaborative.pt"
    load_encoder_weights(model, load_checkpoint(checkpoint_path, device), strict=False)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-2)),
        momentum=float(settings.get("momentum", 0.9)),
        weight_decay=float(settings.get("weight_decay", 1e-3)),
    )
    weights = [float(v) for v in get(config, "model.layer_weights", [0.1, 0.27, 0.189, 0.441])]
    temperature = float(get(config, "model.temperature", 0.1))
    episodes = int(settings.get("episodes", 1200))
    episodes_per_batch = int(settings.get("episodes_per_batch", 4))
    output = resolve_path(config, get(config, "output_dir", "outputs")) / "meta.pt"
    history = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    progress = tqdm(range(1, episodes + 1), desc="Meta-training")
    for episode_index in progress:
        episode = factory.sample()
        support = episode.support_images.to(device)
        query = episode.query_images.to(device)
        support_labels = episode.support_labels.to(device)
        query_labels = episode.query_labels.to(device)
        support_embeddings = model(support, return_embeddings=True)
        query_embeddings = model(query, return_embeddings=True)
        logits, loss = multilevel_episode(
            support_embeddings, support_labels, query_embeddings, query_labels, weights, temperature
        )
        assert loss is not None
        (loss / episodes_per_batch).backward()
        if episode_index % episodes_per_batch == 0 or episode_index == episodes:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        accuracy = (logits.argmax(1) == query_labels).float().mean().item()
        history.append({"episode": episode_index, "loss": loss.item(), "accuracy": accuracy})
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.3f}")
        if episode_index % int(settings.get("save_every", 100)) == 0 or episode_index == episodes:
            save_checkpoint(
                output,
                stage="meta_train",
                episode=episode_index,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                class_names=source.class_names,
                config=config,
                history=history,
            )
    return output


@torch.inference_mode()
def evaluate(config: dict[str, Any], device: torch.device, checkpoint: str | Path | None = None) -> Path:
    target = collection_from_config(config, "target")
    settings = get(config, "evaluation", {})
    checkpoint_path = Path(checkpoint) if checkpoint else resolve_path(
        config, get(config, "output_dir", "outputs")
    ) / "meta.pt"
    state = load_checkpoint(checkpoint_path, device)
    source_classes = state.get("class_names", [])
    model = build_model(max(len(source_classes), 1), get(config, "model.architecture", "resnet10")).to(device)
    load_encoder_weights(model, state, strict=False)
    model.eval()
    weights = [float(v) for v in get(config, "model.layer_weights", [0.1, 0.27, 0.189, 0.441])]
    temperature = float(get(config, "model.temperature", 0.1))
    image_size = int(get(config, "data.image_size", 224))
    generated_per_class = int(settings.get("generated_per_class", 1))
    generated_pool = None
    if generated_per_class:
        generated_root = resolve_path(config, get(config, "data.generated.folder", "data/generated"))
        qc_value = get(config, "data.generated.qc_manifest")
        generated_pool = load_generated_pool(
            generated_root,
            target.class_names,
            resolve_path(config, qc_value) if qc_value else None,
        )
    all_results = []
    seed = int(get(config, "seed", 1))
    for shot in settings.get("shots", [1, 5, 20, 50]):
        factory = EpisodeFactory(
            target,
            ways=int(settings.get("ways", 5)),
            shots=int(shot),
            queries=int(settings.get("queries", 15)),
            image_size=image_size,
            train=False,
            chestx_style=False,
            seed=seed + int(shot) * 1009,
        )
        generator = random.Random(seed + int(shot) * 2027)
        accuracies = []
        for _ in tqdm(range(int(settings.get("episodes", 600))), desc=f"Evaluation {shot}-shot"):
            episode = factory.sample()
            if generated_pool is not None:
                episode = add_generated_support(
                    episode, generated_pool, generated_per_class, image_size, generator
                )
            support_embeddings = model(episode.support_images.to(device), return_embeddings=True)
            query_embeddings = model(episode.query_images.to(device), return_embeddings=True)
            logits, _ = multilevel_episode(
                support_embeddings,
                episode.support_labels.to(device),
                query_embeddings,
                None,
                weights,
                temperature,
            )
            predicted = logits.argmax(1).cpu()
            accuracies.append((predicted == episode.query_labels).float().mean().item())
        mean, ci95 = mean_confidence_interval(accuracies)
        all_results.append(
            {
                "ways": int(settings.get("ways", 5)),
                "shots": int(shot),
                "queries_per_class": int(settings.get("queries", 15)),
                "episodes": len(accuracies),
                "generated_support_per_class": generated_per_class,
                "accuracy": mean,
                "ci95": ci95,
                "accuracy_percent": 100 * mean,
                "ci95_percent": 100 * ci95,
            }
        )
    output = resolve_path(config, get(config, "output_dir", "outputs")) / "evaluation.json"
    write_json(output, {"checkpoint": str(checkpoint_path), "results": all_results, "config": config})
    return output
