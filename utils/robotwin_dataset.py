"""Local RoboTwin v3 reader; use LeRobot's video decoder with episode timestamps."""

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from utils.robotwin import ROBOTWIN_CONTRACT, ROBOTWIN_JOINT_CONTRACT, ROBOTWIN_CAMERA_MAP, read_robotwin_info


class RobotWinDemoReader:
    def __init__(self, dataset_root, *, video_backend="pyav"):
        self.dataset_root = Path(dataset_root)
        self.info = read_robotwin_info(self.dataset_root)
        self.vector_width = self.info["features"]["action"]["shape"][0]
        self.action_contract = ROBOTWIN_CONTRACT if self.vector_width == 20 else ROBOTWIN_JOINT_CONTRACT
        self.video_backend = video_backend
        self._tables = OrderedDict()
        self._ranges = []
        offset = 0
        for path in sorted((self.dataset_root / "data").glob("chunk-*/file-*.parquet")):
            rows = pq.ParquetFile(path).metadata.num_rows
            self._ranges.append((offset, offset + rows, path))
            offset += rows
        if offset != self.info["total_frames"] or not self._ranges:
            raise ValueError(f"RoboTwin parquet rows ({offset}) do not match total_frames")
        self._file_starts = [start for start, _, _ in self._ranges]
        metadata = []
        for path in sorted((self.dataset_root / "meta/episodes").glob("chunk-*/file-*.parquet")):
            columns = [key for key in pq.read_schema(path).names if not key.startswith("stats/")]
            metadata.append(pd.read_parquet(path, columns=columns))
        if not metadata:
            raise FileNotFoundError("Missing RoboTwin episode metadata")
        self.episodes = pd.concat(metadata, ignore_index=True).sort_values("dataset_from_index").reset_index(drop=True)
        starts = self.episodes["dataset_from_index"].to_numpy()
        ends = self.episodes["dataset_to_index"].to_numpy()
        if (len(self.episodes) != self.info["total_episodes"] or starts[0] != 0 or ends[-1] != offset
                or not np.array_equal(starts[1:], ends[:-1])
                or not np.array_equal(ends - starts, self.episodes["length"].to_numpy())):
            raise ValueError("RoboTwin episode ranges do not cover the data consistently")
        self._episode_starts = starts
        tasks = pd.read_parquet(self.dataset_root / "meta/tasks.parquet")
        self.tasks = {int(row["task_index"]): str(row.get("task", index)) for index, row in tasks.iterrows()}
        self._validate_videos()

    def _validate_videos(self):
        for key in ROBOTWIN_CAMERA_MAP:
            if self.info["features"].get(key, {}).get("dtype") != "video":
                raise ValueError(f"Missing RoboTwin video feature {key}")
            columns = [f"videos/{key}/chunk_index", f"videos/{key}/file_index"]
            for chunk, file in self.episodes[columns].drop_duplicates().itertuples(index=False, name=None):
                path = self._video_path(key, chunk, file)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing RoboTwin video (check videos symlink): {path}")

    def _video_path(self, key, chunk, file):
        return self.dataset_root / self.info["video_path"].format(
            video_key=key, chunk_index=int(chunk), file_index=int(file)
        )

    def _episode(self, index):
        position = bisect_right(self._episode_starts, index) - 1
        if position < 0 or index >= int(self.episodes.iloc[position]["dataset_to_index"]):
            raise IndexError(f"RoboTwin index out of bounds: {index}")
        return self.episodes.iloc[position]

    def _table_at(self, index):
        self._episode(index)
        start, end, path = self._ranges[bisect_right(self._file_starts, index) - 1]
        if path not in self._tables:
            table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp", "observation.state", "action"])
            indices = table["index"].to_numpy()
            if not np.array_equal(indices, np.arange(start, end)):
                raise ValueError(f"Noncontiguous global indices in {path}")
            self._tables[path] = table
            # Bound memory per DDP process; never materialize all six million rows as Python objects.
            if len(self._tables) > 2:
                self._tables.popitem(last=False)
        self._tables.move_to_end(path)
        return self._tables[path], start, end

    def read_row(self, index):
        table, start, _ = self._table_at(index)
        return table.slice(index - start, 1).to_pylist()[0]

    def read_observation(self, index):
        from lerobot.datasets.video_utils import decode_video_frames

        episode = self._episode(index)
        row = self.read_row(index)
        if int(row["episode_index"]) != int(episode["episode_index"]):
            raise ValueError("RoboTwin row/episode metadata disagree")
        state = torch.tensor(row["observation.state"], dtype=torch.float32)
        self._validate_vector(state, action=False)
        observation = {"observation.state": state, "task": self.tasks[int(row["task_index"])],
                       "task_index": int(row["task_index"]), "episode_index": int(row["episode_index"]),
                       "index": int(row["index"])}
        for key in ROBOTWIN_CAMERA_MAP:
            prefix = f"videos/{key}/"
            path = self._video_path(key, episode[prefix + "chunk_index"], episode[prefix + "file_index"])
            timestamp = float(row["timestamp"]) + float(episode[prefix + "from_timestamp"])
            observation[key] = decode_video_frames(path, [timestamp], tolerance_s=1e-4,
                                                   backend=self.video_backend, return_uint8=False)[0]
        return observation

    def _validate_vector(self, value, *, action):
        if value.shape[-1] != self.vector_width or not torch.isfinite(value).all():
            raise ValueError(f"Expected finite RoboTwin vectors of width {self.vector_width}")
        if self.vector_width == 14:
            return
        grippers = value[..., [9, 19]]
        if (grippers < (-1e-5 if action else -1.00001)).any() or (grippers > 1.00001).any():
            raise ValueError("RoboTwin gripper values disagree with the EE6D v2 contract")

    def read_action_window(self, index, episode_to_index, horizon):
        episode = self._episode(index)
        if horizon <= 0 or int(episode["dataset_to_index"]) != episode_to_index:
            raise ValueError("Invalid action horizon or episode end")
        stop = min(index + horizon, episode_to_index)
        parts = []
        while index < stop:
            table, start, end = self._table_at(index)
            count = min(stop, end) - index
            piece = table.slice(index - start, count)
            if not np.all(piece["episode_index"].to_numpy() == int(episode["episode_index"])):
                raise ValueError("Action window crossed an episode boundary")
            parts.extend(piece["action"].to_pylist())
            index += count
        actions = torch.tensor(parts, dtype=torch.float32)
        self._validate_vector(actions, action=True)
        if len(actions) < horizon:
            actions = torch.cat([actions, actions[-1:].expand(horizon - len(actions), -1)])
        return actions

    def sample_pair_indices(self, *, rng, horizon, execute_steps, batch_size):
        if not 0 < execute_steps < horizon or batch_size <= 0:
            raise ValueError("Expected 0<K<H and batch_size>0")
        valid = self.episodes[self.episodes["length"] >= horizon + execute_steps]
        if valid.empty:
            raise ValueError(f"No episode is long enough for H={horizon}, K={execute_steps}")
        pairs = []
        for _ in range(batch_size):
            episode = valid.iloc[int(rng.integers(len(valid)))]
            end = int(episode["dataset_to_index"])
            old = int(rng.integers(int(episode["dataset_from_index"]), end - horizon - execute_steps + 1))
            pairs.append((old, old + execute_steps, end))
        return pairs
