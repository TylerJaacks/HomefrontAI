"""Reads HomefrontImitation recordings (.hfrec, written by Recording/RecordingWriter.cs).

    rec = hfrec.read("fl-farm_20261106_201500.hfrec")
    rec.samples["loc"], rec.samples["feat"], rec.kills, rec.strings, rec.soldiers

Little-endian. Header: "HFRC", format version u16, feature version u16, feature count u16, sample seconds f32,
start unix i64, then u16-length UTF-8 strings: map, game mode, feature schema. Records start with a tag byte:
1 sample, 2 kill, 3 string table entry, 4 soldier first seen, 5 end of recording.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = 0x43524648  # "HFRC"
FORMAT_VERSION = 1
NO_PID = 0xFFFF

# SampleFlags (RecordingWriter.cs)
BOT, CROUCHED, SPRINTING, FALLING, LADDER, FIRING, JUMPED, HAS_ENEMY, WALKING = (1 << i for i in range(9))

KILL_DTYPE = np.dtype([("t", "<f4"), ("killer", "<u2"), ("victim", "<u2"), ("killer_loc", "<f4", 3),
                       ("victim_loc", "<f4", 3), ("damage", "<u2")])


def sample_dtype(features: int) -> np.dtype:
    return np.dtype([
        ("t", "<f4"), ("pid", "<u2"), ("team", "i1"), ("flags", "<u2"), ("health", "<f4"),
        ("loc", "<f4", 3), ("vel", "<f4", 3), ("acc", "<f4", 3),
        ("accel_rate", "<f4"), ("ground_speed", "<f4"), ("yaw", "<f4"), ("pitch", "<f4"),
        ("floor_z", "<f4"), ("frame", "<f4", 2), ("enemy", "<u2"), ("weapon", "<u2"),
        ("feat", "<f4", features),
    ])


@dataclass
class Recording:
    path: str
    map: str
    mode: str
    schema: str
    feature_version: int
    feature_count: int
    sample_seconds: float
    start: int
    samples: np.ndarray
    kills: np.ndarray
    strings: dict = field(default_factory=dict)     # id -> str
    soldiers: dict = field(default_factory=dict)    # pid -> (team, is_bot)
    complete: bool = False                          # ended with the end tag (not cut off by a crash)

    @property
    def feature_names(self) -> list[str]:
        return self.schema.split(":", 1)[1].split(",") if ":" in self.schema else []

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class FormatError(Exception):
    pass


def read(path: str) -> Recording:
    data = open(path, "rb").read()
    if len(data) < 22:
        raise FormatError(f"{path}: too short")
    magic, version, fver, fcount, sample_s, start = struct.unpack_from("<IHHHfq", data, 0)
    if magic != MAGIC:
        raise FormatError(f"{path}: not a HomefrontImitation recording")
    if version != FORMAT_VERSION:
        raise FormatError(f"{path}: format version {version}, this reader knows {FORMAT_VERSION}")
    pos = 22

    def string(p):
        (n,) = struct.unpack_from("<H", data, p)
        return data[p + 2:p + 2 + n].decode("utf-8", "replace"), p + 2 + n

    map_name, pos = string(pos)
    mode, pos = string(pos)
    schema, pos = string(pos)

    sdt = sample_dtype(fcount)
    ssize, ksize = sdt.itemsize, KILL_DTYPE.itemsize
    sample_at, kill_at = [], []
    strings, soldiers = {}, {}
    complete = False
    end = len(data)
    while pos < end:
        tag = data[pos]
        pos += 1
        if tag == 1:
            if pos + ssize > end:
                break
            sample_at.append(pos)
            pos += ssize
        elif tag == 2:
            if pos + ksize > end:
                break
            kill_at.append(pos)
            pos += ksize
        elif tag == 3:
            if pos + 4 > end:
                break
            (sid,) = struct.unpack_from("<H", data, pos)
            s, pos = string(pos + 2)
            strings[sid] = s
        elif tag == 4:
            if pos + 4 > end:
                break
            pid, team, bot = struct.unpack_from("<HbB", data, pos)
            soldiers[pid] = (team, bool(bot))
            pos += 4
        elif tag == 5:
            complete = True
            break
        else:
            # A server crash can leave a torn last record; anything else is corruption. Keep what came before.
            break

    buf = np.frombuffer(data, dtype=np.uint8)

    def gather(offsets, dtype, chunk=50000):
        out = np.zeros(len(offsets), dtype=dtype)
        if not offsets:
            return out
        raw = out.view(np.uint8).reshape(len(offsets), dtype.itemsize)
        cols = np.arange(dtype.itemsize, dtype=np.int64)
        offs = np.asarray(offsets, dtype=np.int64)
        for a in range(0, len(offs), chunk):   # a bounded index array, not one per byte of the whole file
            raw[a:a + chunk] = buf[offs[a:a + chunk, None] + cols]
        return out

    return Recording(path=path, map=map_name, mode=mode, schema=schema, feature_version=fver,
                     feature_count=fcount, sample_seconds=sample_s, start=start,
                     samples=gather(sample_at, sdt), kills=gather(kill_at, KILL_DTYPE),
                     strings=strings, soldiers=soldiers, complete=complete)


def find(paths: list[str]) -> list[str]:
    """Every .hfrec under the given files/folders, oldest first."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out += [os.path.join(root, f) for f in files if f.endswith(".hfrec")]
        elif p.endswith(".hfrec"):
            out.append(p)
    return sorted(set(out))


def move_class(fwd: np.ndarray, right: np.ndarray, still: float = 0.3) -> np.ndarray:
    """Features.MoveClass: 0 still, 1..8 = 0, 45, ..., 315 degrees from the frame's forward toward its right.
    Rounds half to even, like .NET's MathF.Round."""
    a = np.arctan2(right, fwd)
    i = np.round(a / (np.pi / 4)).astype(np.int64)
    k = np.mod(i, 8) + 1
    return np.where(np.hypot(fwd, right) < still, 0, k)


def in_frame(vec_xy: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features.ToFrame: flat world vectors -> (forward, right); right = (-frame.y, frame.x)."""
    fwd = vec_xy[:, 0] * frame[:, 0] + vec_xy[:, 1] * frame[:, 1]
    right = vec_xy[:, 0] * -frame[:, 1] + vec_xy[:, 1] * frame[:, 0]
    return fwd, right
