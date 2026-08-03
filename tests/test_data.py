from pathlib import Path

from PIL import Image

from dsga.data import EpisodeFactory, ImageCollection


def test_episode_support_and_query_are_disjoint(tmp_path: Path):
    for class_name in ("a", "b", "c"):
        directory = tmp_path / class_name
        directory.mkdir()
        for index in range(5):
            Image.new("RGB", (32, 32), color=(index * 30, 10, 20)).save(directory / f"{index}.png")
    collection = ImageCollection.from_folder(tmp_path)
    episode = EpisodeFactory(collection, ways=2, shots=1, queries=2, image_size=32, seed=3).sample()
    assert set(episode.support_paths).isdisjoint(episode.query_paths)
    assert episode.support_images.shape[0] == 2
    assert episode.query_images.shape[0] == 4

