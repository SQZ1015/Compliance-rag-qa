# -*- coding: utf-8 -*-
"""下载 BGE-M3 到本地目录：小文件单连接下载，权重文件多线程 Range 分片并行下载。

环境变量：
  HF_DOWNLOAD_BASE  默认 https://huggingface.co（失败可设 https://hf-mirror.com）
  LOCAL_MODEL_DIR   默认 D:/hf_cache/local/bge-m3
"""
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = os.environ.get("HF_DOWNLOAD_BASE", "https://huggingface.co").rstrip("/")
REPO = os.environ.get("HF_REPO", "BAAI/bge-small-zh-v1.5")
OUT = Path(os.environ.get("LOCAL_MODEL_DIR", r"D:\hf_cache\local\bge-small-zh-v1.5"))
BIG_FILE = os.environ.get("HF_BIG_FILE", "model.safetensors")

SMALL_FILES = [f for f in os.environ.get("HF_SMALL_FILES", "").split(",") if f] or [
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "1_Pooling/config.json",
]
CHUNK = 8 * 1024 * 1024
WORKERS = 16


def url(path):
    return f"{BASE}/{REPO}/resolve/main/{path}"


def fetch(path, timeout=90, retries=4):
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url(path), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last


def total_size(path):
    req = urllib.request.Request(url(path), headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
    return int(cr.rsplit("/", 1)[-1])


def download_range(path, start, end, retries=5):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url(path), headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return start, r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"source={BASE}")
    for path in SMALL_FILES:
        dest = OUT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = fetch(path)
        dest.write_bytes(data)
        print(f"  ok {path} ({len(data)} B)")

    total = total_size(BIG_FILE)
    print(f"big file {BIG_FILE} total={total/1024/1024:.0f} MB, workers={WORKERS}")
    out = OUT / BIG_FILE
    with out.open("wb") as f:
        f.truncate(total)

    done = 0
    t0 = time.time()
    ranges = [(s, min(s + CHUNK - 1, total - 1)) for s in range(0, total, CHUNK)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(download_range, BIG_FILE, s, e) for s, e in ranges]
        with out.open("r+b") as f:
            for fut in as_completed(futs):
                start, data = fut.result()
                f.seek(start)
                f.write(data)
                done += len(data)
                if done % (32 * 1024 * 1024) < CHUNK or done >= total:
                    pct = done / total * 100
                    el = time.time() - t0
                    print(f"  {pct:5.1f}%  {done/1024/1024:.0f}/{total/1024/1024:.0f} MB  {done/el/1024/1024:.1f} MB/s")
    print(f"done -> {out} ({total/1024/1024:.0f} MB)")


if __name__ == "__main__":
    main()
