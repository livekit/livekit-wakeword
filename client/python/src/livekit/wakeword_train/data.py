"""Training data: the wake word's bench clips, background speech streamed from the Hugging Face
Hub (never stored), and the bench's train noise and room impulse response pools."""

from __future__ import annotations

import fnmatch
import io
import json
import logging
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000
CACHE = Path.home() / ".cache" / "livekit-wakeword"

_FLEURS = "en_us es_419 de_de fr_fr it_it pt_br ru_ru hi_in ja_jp ko_kr cmn_hans_cn vi_vn ar_eg"
_CV = "en es de fr it pt ru hi ja ko vi ar tr nl"
NEGATIVE_SPEECH: list[tuple[str, list[str]]] = [
    ("openslr/librispeech_asr", ["clean/train.100/*.parquet", "other/train.500/*.parquet"]),
    (
        "google/fleurs",
        [f"parquet-data/{x}/train-*.parquet" for x in (_FLEURS + " tr_tr nl_nl pl_pl").split()],
    ),
    (
        "fixie-ai/common_voice_17_0",
        [f"{x}/train*.parquet" for x in _CV.split()]
        + [f"{x}/train/*/train-*.parquet" for x in ("ko", "vi", "hi")],
    ),
]
TRAIN_NOISE = ["musan_noise", "dns_noise"]
TRAIN_RIRS = ["mit_rir"]


@dataclass
class ClipSet:
    """Clips packed into one int16 array, with the wake word (or utterance) span of each."""

    audio: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets)

    def clip(self, i: int) -> np.ndarray:
        return np.asarray(self.audio[self.offsets[i] : self.offsets[i] + self.lengths[i]])


_KEYS = ("audio", "offsets", "lengths", "starts", "ends")
_CATS = ("positive", "adversarial", "speech")


def _concat(sets: list[ClipSet]) -> ClipSet:
    offs, base = [], 0
    for s in sets:
        offs.append(np.asarray(s.offsets) + base)
        base += len(s.audio)
    return ClipSet(
        np.concatenate([np.asarray(s.audio) for s in sets]),
        np.concatenate(offs),
        *(np.concatenate([np.asarray(getattr(s, k)) for s in sets]) for k in _KEYS[2:]),
    )


def load_clipsets(dataset: str | Path | list[str]) -> dict[str, ClipSet]:
    """A bench dataset's (dir or HF repo; a list is merged) train split as a ClipSet per category,
    cached as mmap-able .npy."""
    if isinstance(dataset, list):
        parts = [load_clipsets(d) for d in dataset]
        return {c: _concat([p[c] for p in parts if len(p[c])] or [parts[0][c]]) for c in _CATS}
    from livekit_wakeword_bench import WakeWordData

    cdir = CACHE / "clipsets" / str(dataset).strip("/").replace("/", "_")
    if all((cdir / c / "ends.npy").exists() for c in _CATS):
        return {
            c: ClipSet(
                *(
                    np.load(cdir / c / f"{k}.npy", mmap_mode="r" if k == "audio" else None)
                    for k in _KEYS
                )
            )
            for c in _CATS
        }
    acc: dict[str, dict[str, list[Any]]] = {c: {k: [] for k in _KEYS} for c in _CATS}
    pos = dict.fromkeys(_CATS, 0)
    for s in WakeWordData(str(dataset), split="train"):
        cat = s.meta["category"]
        if cat not in acc:
            continue
        on, off = (
            (s.meta["phrase_onset_s"], s.meta["phrase_offset_s"])
            if cat == "positive"
            else (s.meta["onset_s"], s.meta["offset_s"])
        )
        pcm = (np.clip(s.audio, -1, 1) * 32767).astype(np.int16)
        a = acc[cat]
        a["audio"].append(pcm)
        a["offsets"].append(pos[cat])
        a["lengths"].append(len(pcm))
        a["starts"].append(int(on * SR))
        a["ends"].append(min(len(pcm), int(off * SR)))
        pos[cat] += len(pcm)
    out = {}
    tmp = cdir.with_name(cdir.name + f".tmp{os.getpid()}")  # built aside, then renamed
    for c in _CATS:
        a = (
            acc[c]
            if acc[c]["audio"]
            else {"audio": [np.zeros(0, np.int16)], **{k: [] for k in _KEYS[1:]}}
        )
        cs = ClipSet(np.concatenate(a["audio"]), *(np.array(a[k], np.int64) for k in _KEYS[1:]))
        (tmp / c).mkdir(parents=True, exist_ok=True)
        for k in _KEYS:
            np.save(tmp / c / f"{k}.npy", getattr(cs, k))
        out[c] = cs
    shutil.rmtree(cdir, ignore_errors=True)
    os.replace(tmp, cdir)
    return out


def _bench_sources(names: list[str]) -> list[Path]:
    from livekit_wakeword_bench.sources import prepare_source, source_dir

    cache = CACHE / "sources"
    files: list[Path] = []
    for n in names:
        items = prepare_source(n, cache)
        files += [source_dir(cache, n) / it.name for it in items]
    return files


def noise_pool() -> np.ndarray:
    """Train-pool noise concatenated into one float16 array (each file normalized)."""
    path = CACHE / "pools" / "noise.npy"
    if path.exists():
        cached: np.ndarray = np.load(path, mmap_mode="r")
        return cached
    import soundfile as sf

    parts = []
    for f in _bench_sources(TRAIN_NOISE):
        x, _ = sf.read(f, dtype="float32", always_2d=True)
        x = x.mean(1)
        rms = float(np.sqrt((x**2).mean() + 1e-12))
        if rms > 1e-5:
            parts.append((x / rms * 0.1).astype(np.float16))
    arr = np.concatenate(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def rir_pool(max_len: int = SR) -> np.ndarray:
    """(N, max_len) RIRs, direct path at index 0, unit energy."""
    path = CACHE / "pools" / "rirs.npy"
    if path.exists():
        cached: np.ndarray = np.load(path)
        return cached
    import soundfile as sf

    out = []
    for f in _bench_sources(TRAIN_RIRS):
        x, _ = sf.read(f, dtype="float32", always_2d=True)
        x = x[:, 0]
        pk = int(np.argmax(np.abs(x)))
        x = x[max(0, pk - 16) :][:max_len]
        x = np.pad(x, (0, max_len - len(x)))
        out.append((x / (np.sqrt((x**2).sum()) + 1e-8)).astype(np.float32))
    arr = np.stack(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def _decode(blob: object) -> np.ndarray | None:
    import soundfile as sf
    import soxr

    if not isinstance(blob, dict):
        return None
    x: np.ndarray
    if blob.get("bytes"):
        x, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=True)
        x = x.mean(1)
    elif blob.get("array") is not None:
        x, sr = np.asarray(blob["array"], dtype=np.float32), int(blob["sampling_rate"])
    else:
        return None
    if sr != SR:
        resampled: np.ndarray = soxr.resample(x, sr, SR)
        return resampled
    return x


BULKY_ROW_GROUP = 64e6  # bytes; shards with larger row groups are skipped while others remain
PER_GROUP = 24  # utterances kept per row group read


class StreamPool:
    """A background thread filling a rolling int16 RAM buffer with speech streamed from the Hub
    (a few random utterances per random row group, over HTTP range requests); `crop()` samples
    windows from it."""

    def __init__(
        self,
        files: list[tuple[str, str]],
        buffer_s: float = 1800.0,
        seed: int = 0,
        min_fill_s: float = 300.0,
        refresh_s: float = 600.0,
    ) -> None:
        self.files = list(files)
        self.buf = np.zeros(int(buffer_s * SR), np.int16)
        self.head = self.filled = 0
        self.min_fill = min(int(min_fill_s * SR), len(self.buf))
        self.refresh_s = refresh_s
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self._stopped = False
        self._start()

    def stop(self, join: bool = False, timeout: float = 60.0) -> None:
        """Stop streaming; the buffer stays readable."""
        self._stopped = True
        if join:
            self.thread.join(timeout)

    def _start(self) -> None:
        self.thread = threading.Thread(target=self._run_forever, daemon=True)
        self.thread.start()

    def _run_forever(self) -> None:
        while not self._stopped:  # never let the filler die: it would stall a data worker
            try:
                self._run()
            except BaseException as e:  # noqa: BLE001
                logger.warning("speech stream restarting after: %r", e)
                time.sleep(5)

    def _write(self, pcm: np.ndarray) -> None:
        N = len(self.buf)
        pcm = pcm[-N:]
        n = len(pcm)
        with self.lock:
            end = self.head + n
            if end <= N:
                self.buf[self.head : end] = pcm
            else:
                k = N - self.head
                self.buf[self.head :] = pcm[:k]
                self.buf[: n - k] = pcm[k:]
            self.head = end % N
            self.filled = min(N, self.filled + n)

    def _run(self) -> None:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        opened: dict[tuple[str, str], Any] = {}
        files = self.files[:]
        while not self._stopped:
            repo, f = self.rng.choice(files)
            try:
                if (repo, f) not in opened:
                    fh = fs.open(f"datasets/{repo}/{f}", "rb", block_size=2**20)
                    pf = pq.ParquetFile(fh)
                    md = pf.metadata
                    if fh.size / md.num_row_groups > BULKY_ROW_GROUP and len(files) > 1:
                        files = [x for x in files if x != (repo, f)]
                        continue
                    opened[(repo, f)] = pf
                pf = opened[(repo, f)]
                rg = self.rng.randrange(pf.metadata.num_row_groups)
                rows = pf.read_row_group(rg, columns=["audio"]).column("audio").to_pylist()
                for row in self.rng.sample(rows, min(PER_GROUP, len(rows))):
                    x = _decode(row)
                    if x is None or len(x) < SR // 2:
                        continue
                    self._write((np.clip(x, -1, 1) * 32767).astype(np.int16))
                    if self.filled >= len(self.buf) and self.refresh_s > 0:
                        time.sleep(len(x) / len(self.buf) * self.refresh_s)
            except Exception as e:
                logger.warning("stream %s/%s: %s", repo, f, e)
                opened.pop((repo, f), None)
                time.sleep(5)

    def crop(self, rng: np.random.Generator, length: int, max_wait_s: float = 180.0) -> np.ndarray:
        t0 = time.time()
        while self.filled < self.min_fill and time.time() - t0 < max_wait_s:
            if not self.thread.is_alive() and not self._stopped:
                self._start()
            time.sleep(0.5)
        if self.filled < self.min_fill:
            logger.warning("speech stream slow: %.0f s buffered", self.filled / SR)
        with self.lock:
            if self.filled < length:
                out = np.zeros(length, np.int16)
                out[: self.filled] = self.buf[: self.filled]
                return out
            pos = int(rng.integers(0, max(1, self.filled - length)))
            if self.filled < len(self.buf):
                return self.buf[pos : pos + length].copy()
            idx = (self.head + pos + np.arange(length)) % len(self.buf)
            return self.buf[idx].copy()


def negative_speech_files() -> list[tuple[str, str]]:
    """Resolve NEGATIVE_SPEECH globs to concrete shard paths (cached listing)."""
    path = CACHE / "negative_speech_files.json"
    if path.exists():
        return [tuple(x) for x in json.loads(path.read_text())]
    from huggingface_hub import HfApi

    api = HfApi()
    out: list[tuple[str, str]] = []
    for repo, globs in NEGATIVE_SPEECH:
        files = api.list_repo_files(repo, repo_type="dataset")
        for g in globs:
            out += [(repo, f) for f in sorted(files) if fnmatch.fnmatch(f, g)][:8]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out
