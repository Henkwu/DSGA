from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .augmentation import ChestXStyleTransform


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str
    label: int


class ImageCollection:
    """An image collection backed by a class-folder tree or a CSV manifest."""

    def __init__(self, samples: list[Sample], class_names: list[str]):
        if not samples:
            raise ValueError("The dataset contains no images")
        self.samples = samples
        self.class_names = class_names
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.indices_by_class: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
        for index, sample in enumerate(samples):
            self.indices_by_class[sample.label].append(index)

    @classmethod
    def from_folder(cls, root: str | Path) -> "ImageCollection":
        root = Path(root)
        class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
        samples = []
        for label, name in enumerate(class_names):
            for path in sorted((root / name).rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(Sample(path.resolve(), name, label))
        return cls(samples, class_names)

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        root: str | Path | None = None,
        split: str | None = None,
    ) -> "ImageCollection":
        manifest = Path(manifest)
        root_path = Path(root).resolve() if root else manifest.parent.resolve()
        rows: list[tuple[Path, str]] = []
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"path", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest must contain columns {sorted(required)}")
            for row in reader:
                if split and row.get("split") and row["split"] != split:
                    continue
                path = Path(row["path"])
                if not path.is_absolute():
                    path = root_path / path
                rows.append((path.resolve(), row["label"].strip()))
        class_names = sorted({label for _, label in rows})
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        samples = [Sample(path, name, class_to_idx[name]) for path, name in rows]
        missing = [str(s.path) for s in samples if not s.path.is_file()]
        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(f"Manifest references missing images (first five):\n{preview}")
        return cls(samples, class_names)

    @classmethod
    def load(
        cls,
        folder: str | Path | None = None,
        manifest: str | Path | None = None,
        root: str | Path | None = None,
        split: str | None = None,
    ) -> "ImageCollection":
        if bool(folder) == bool(manifest):
            raise ValueError("Set exactly one of folder or manifest")
        return cls.from_folder(folder) if folder else cls.from_manifest(manifest, root, split)

    def open(self, index: int) -> Image.Image:
        with Image.open(self.samples[index].path) as image:
            return image.convert("RGB")


def build_transform(image_size: int = 224, train: bool = False) -> Callable[[Image.Image], torch.Tensor]:
    operations: list[Callable] = []
    if train:
        operations += [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    else:
        operations += [transforms.Resize((image_size, image_size))]
    operations += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(operations)


class CollaborativeDataset(Dataset):
    """Returns original and domain-specific transformed views with a shared label."""

    def __init__(
        self,
        collection: ImageCollection,
        image_size: int = 224,
        clip_limit: float = 4.0,
        tile_grid: int = 8,
        blur_kernel: int = 13,
    ):
        self.collection = collection
        self.image_transform = build_transform(image_size, train=True)
        self.style_transform = ChestXStyleTransform(clip_limit, tile_grid, blur_kernel)

    def __len__(self) -> int:
        return len(self.collection.samples)

    def __getitem__(self, index: int):
        image = self.collection.open(index)
        label = self.collection.samples[index].label
        return self.image_transform(image), self.image_transform(self.style_transform(image)), label


@dataclass
class Episode:
    support_images: torch.Tensor
    support_labels: torch.Tensor
    query_images: torch.Tensor
    query_labels: torch.Tensor
    class_names: list[str]
    support_paths: list[str]
    query_paths: list[str]


class EpisodeFactory:
    def __init__(
        self,
        collection: ImageCollection,
        ways: int,
        shots: int,
        queries: int,
        image_size: int = 224,
        train: bool = False,
        chestx_style: bool = False,
        seed: int = 0,
    ):
        self.collection = collection
        self.ways = ways
        self.shots = shots
        self.queries = queries
        self.transform = build_transform(image_size, train=train)
        self.style_transform = ChestXStyleTransform() if chestx_style else None
        self.rng = random.Random(seed)
        eligible = [
            label for label, indices in collection.indices_by_class.items()
            if len(indices) >= shots + queries
        ]
        if len(eligible) < ways:
            raise ValueError(
                f"Need {ways} classes with at least {shots + queries} images; found {len(eligible)}"
            )
        self.eligible_classes = eligible

    def _tensor(self, index: int) -> torch.Tensor:
        image = self.collection.open(index)
        if self.style_transform:
            image = self.style_transform(image)
        return self.transform(image)

    def sample(self) -> Episode:
        chosen = self.rng.sample(self.eligible_classes, self.ways)
        support_images, query_images = [], []
        support_labels, query_labels = [], []
        support_paths, query_paths, class_names = [], [], []
        used_paths: set[Path] = set()
        for episodic_label, original_label in enumerate(chosen):
            candidates = [
                index for index in self.collection.indices_by_class[original_label]
                if self.collection.samples[index].path not in used_paths
            ]
            required = self.shots + self.queries
            if len(candidates) < required:
                raise ValueError(
                    "Cannot construct an image-disjoint episode. Use a single-label manifest "
                    "or reduce shots/queries."
                )
            indices = self.rng.sample(candidates, required)
            used_paths.update(self.collection.samples[index].path for index in indices)
            support_indices, query_indices = indices[: self.shots], indices[self.shots :]
            support_images.extend(self._tensor(i) for i in support_indices)
            query_images.extend(self._tensor(i) for i in query_indices)
            support_labels.extend([episodic_label] * self.shots)
            query_labels.extend([episodic_label] * self.queries)
            support_paths.extend(str(self.collection.samples[i].path) for i in support_indices)
            query_paths.extend(str(self.collection.samples[i].path) for i in query_indices)
            class_names.append(self.collection.class_names[original_label])
        return Episode(
            torch.stack(support_images),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_images),
            torch.tensor(query_labels, dtype=torch.long),
            class_names,
            support_paths,
            query_paths,
        )


def load_generated_pool(
    root: str | Path,
    class_names: list[str],
    qc_manifest: str | Path | None = None,
) -> dict[str, list[Path]]:
    root = Path(root)
    accepted: set[Path] | None = None
    if qc_manifest:
        accepted = set()
        manifest_path = Path(qc_manifest)
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if str(row.get("accepted", "")).strip().lower() in {"1", "true", "yes", "y"}:
                    path = Path(row["path"])
                    accepted.add((manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve())
    pool: dict[str, list[Path]] = {}
    for class_name in class_names:
        directory = root / class_name
        paths = sorted(p.resolve() for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if directory.exists() else []
        if accepted is not None:
            paths = [path for path in paths if path in accepted]
        pool[class_name] = paths
    return pool


def add_generated_support(
    episode: Episode,
    generated_pool: dict[str, list[Path]],
    per_class: int,
    image_size: int,
    rng: random.Random,
) -> Episode:
    if per_class <= 0:
        return episode
    transform = build_transform(image_size, train=False)
    images = [image for image in episode.support_images]
    labels = episode.support_labels.tolist()
    paths = list(episode.support_paths)
    for episodic_label, class_name in enumerate(episode.class_names):
        candidates = generated_pool.get(class_name, [])
        if len(candidates) < per_class:
            raise ValueError(f"Class {class_name!r} has {len(candidates)} generated images; need {per_class}")
        for path in rng.sample(candidates, per_class):
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
            labels.append(episodic_label)
            paths.append(str(path))
    return Episode(
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
        episode.query_images,
        episode.query_labels,
        episode.class_names,
        paths,
        episode.query_paths,
    )
