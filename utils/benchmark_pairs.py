from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch


def _decode_image(value: Any) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    if isinstance(value, dict) and "bytes" in value:
        image = Image.open(BytesIO(value["bytes"])).convert("RGB")
    elif isinstance(value, (bytes, bytearray)):
        image = Image.open(BytesIO(value)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image value type: {type(value)!r}")
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


class ParquetDemoReader:
    """Episode-aware reader for local LeRobot-style demonstration datasets."""

    def __init__(self, dataset_root: str | Path, *, benchmark: str) -> None:
        import pandas as pd
        import pyarrow.parquet as pq

        self.dataset_root = Path(dataset_root)
        self.benchmark = benchmark
        self.data_dir = self.dataset_root / "data" / "chunk-000"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Missing parquet data directory: {self.data_dir}")
        self._files: dict[int, Any] = {}
        self._ranges: list[tuple[int, int, int]] = []
        self._data_columns: set[str] = set()
        start = 0
        for path in sorted(self.data_dir.glob("file-*.parquet")):
            parquet_file = pq.ParquetFile(path)
            rows = int(parquet_file.metadata.num_rows)
            file_index = int(path.stem.split("-")[-1])
            self._ranges.append((start, start + rows, file_index))
            start += rows
            self._data_columns.update(parquet_file.schema_arrow.names)
        if not self._ranges:
            raise FileNotFoundError(f"No parquet files found under {self.data_dir}")

        episodes_path = self.dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")
        self.episodes = pd.read_parquet(episodes_path)
        self.tasks = self._load_tasks()
        self._video_dataset = None
        self._image_columns = self._find_inline_images()
        if not self._image_columns:
            self._video_dataset = self._make_video_dataset()

    def _load_tasks(self) -> dict[int, str]:
        import pandas as pd

        path = self.dataset_root / "meta" / "tasks.parquet"
        if path.exists():
            table = pd.read_parquet(path)
            result: dict[int, str] = {}
            for index, row in table.iterrows():
                task_index = int(row.get("task_index", index))
                text = row.get("task", row.get("language_instruction", row.get("task_id", task_index)))
                result[task_index] = str(text)
            return result
        return {}

    def _find_inline_images(self) -> list[str]:
        preferred = [column for column in self._data_columns if column.startswith("observation.images.")]
        if preferred:
            return sorted(preferred)
        if "observation.image" in self._data_columns:
            return ["observation.image"]
        return []

    def _make_video_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_id = {
            "libero": "lerobot/libero",
            "libero_plus": "lerobot/libero_plus",
            "metaworld_mt50": "lerobot/metaworld_mt50",
        }.get(self.dataset_root.name, f"local/{self.dataset_root.name}")
        return LeRobotDataset(
            repo_id=repo_id,
            root=self.dataset_root,
            download_videos=False,
            video_backend="pyav",
        )

    def file_index_for_global_index(self, index: int) -> int:
        for start, end, file_index in self._ranges:
            if start <= index < end:
                return file_index
        raise IndexError(f"Index {index} is outside {self.dataset_root}")

    def _file(self, file_index: int):
        import pandas as pd

        if file_index not in self._files:
            self._files[file_index] = pd.read_parquet(self.data_dir / f"file-{file_index:03d}.parquet")
        return self._files[file_index]

    def read_row(self, index: int) -> dict[str, Any]:
        frame = self._file(self.file_index_for_global_index(index))
        rows = frame[frame["index"] == index]
        if rows.empty:
            raise RuntimeError(f"No row for index={index}")
        return rows.iloc[0].to_dict()

    def _images(self, row: dict[str, Any], index: int) -> dict[str, torch.Tensor]:
        if self._video_dataset is not None:
            item = self._video_dataset[index]
            output: dict[str, torch.Tensor] = {}
            for key in self._video_dataset.features:
                if key.startswith("observation.images."):
                    destination = key
                    if key.endswith(".front"):
                        destination = "observation.images.image"
                    elif key.endswith(".wrist"):
                        destination = "observation.images.image2"
                    output[destination] = item[key].float()
            return output

        output = {}
        for key in self._image_columns:
            destination = key if key.startswith("observation.images.") else "observation.images.image"
            output[destination] = _decode_image(row[key])
        return output

    def read_observation(self, index: int) -> dict[str, Any]:
        row = self.read_row(index)
        task_index = int(row.get("task_index", row.get("task_id", 0)))
        task = self.tasks.get(task_index, str(row.get("task_id", task_index)))
        state = row.get("observation.state")
        if state is None:
            raise KeyError("Dataset row has no observation.state")
        return {
            **self._images(row, index),
            "observation.state": torch.tensor(state, dtype=torch.float32),
            "task": task,
            "task_index": task_index,
            "episode_index": int(row["episode_index"]),
            "index": int(row["index"]),
        }

    def read_action_window(self, index: int, episode_to_index: int, horizon: int) -> torch.Tensor:
        rows: list[dict[str, Any]] = []
        current = index
        end = min(index + horizon, episode_to_index)
        while current < end and len(rows) < horizon:
            frame = self._file(self.file_index_for_global_index(current))
            part = frame[(frame["index"] >= current) & (frame["index"] < end)]
            records = part.to_dict("records")
            if not records:
                raise RuntimeError(f"No action row at index={current}")
            rows.extend(records)
            current = int(records[-1]["index"]) + 1
        actions = torch.stack([torch.tensor(row["action"], dtype=torch.float32) for row in rows])
        if actions.shape[0] < horizon:
            actions = torch.cat([actions, actions[-1:].expand(horizon - actions.shape[0], -1)], dim=0)
        return actions[:horizon]

    def sample_pair_indices(self, *, rng, horizon: int, execute_steps: int, batch_size: int):
        minimum = horizon + execute_steps
        valid = self.episodes[self.episodes["length"] >= minimum]
        if valid.empty:
            raise ValueError(f"No episode has at least H+K={minimum} frames in {self.dataset_root}")
        pairs = []
        for _ in range(batch_size):
            episode = valid.iloc[int(rng.integers(0, len(valid)))]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"]) - minimum
            old_index = int(rng.integers(start, stop + 1))
            pairs.append((old_index, old_index + execute_steps, int(episode["dataset_to_index"])))
        return pairs


def make_reader(dataset_root: str | Path, benchmark: str):
    return ParquetDemoReader(dataset_root, benchmark=benchmark)
