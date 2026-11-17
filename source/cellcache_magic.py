"""
cellcache_magic.py — `%%cache` cell magic that stores a notebook cell's results on disk.

The first run of a cached cell executes it and saves its variables, printed output and plots
in `cache/`. Later runs restore them instead of recomputing, until the cell's code changes.

Usage (in a notebook run from the repository root):

    %load_ext source.cellcache_magic

    %%cache [label] [--force]
    result = expensive_computation()

`label` names the cache files; `--force` recomputes even if a cache exists.

Remarks:
    - Only variables that the cell creates are saved; variables it changes are not.
    - The cache key is the cell's code only. After changing code in `source/`, delete `cache/` (or
      use `--force`) to recompute.
"""

import base64
import hashlib
import os
import sys

import cloudpickle as pickle
from IPython.core.getipython import get_ipython
from IPython.display import Image, display

CACHE_DIR = "cache"


def cache(line, cell):
    """The `%%cache [label] [--force]` cell magic (see the module docstring)."""
    ip = get_ipython()
    if ip is None:
        raise RuntimeError("cellcache_magic requires an active IPython session")
    os.makedirs(CACHE_DIR, exist_ok=True)

    args = line.split()
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    label = args[0] if args else "cell"

    # Cache key: label + hash of the cell's exact source.
    # Any edit to the cell changes the hash -> automatic invalidation.
    code_hash = hashlib.md5(cell.encode()).hexdigest()[:10]
    base = os.path.join(CACHE_DIR, f"{label}_{code_hash}")
    vars_path = base + "_vars.pkl"
    manifest_path = base + "_manifest.pkl"

    # --- Cache hit ---
    if not force and os.path.exists(vars_path) and os.path.exists(manifest_path):
        with open(vars_path, "rb") as f:
            saved_vars = pickle.load(f)
        ip.user_ns.update(saved_vars)

        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)

        # Replay text and images in the exact order they originally occurred.
        for item in manifest:
            if item["type"] == "text":
                print(item["value"], end="")
            elif item["type"] == "image":
                display(Image(filename=item["path"]))

        print(f"[cache hit: '{label}' ({code_hash})]")
        return

    # --- Cache miss: execute for real, recording output as an ordered
    # sequence of events (text chunks and images) so it can be replayed in
    # the exact original order on a future cache hit. ---
    before = set(ip.user_ns.keys())

    manifest = []
    img_idx = 0
    old_stdout = sys.stdout
    old_publish = ip.display_pub.publish

    class _TeeWriter:
        """Writes to real stdout (so output still appears live) while also
        recording each chunk in the manifest, in order."""

        def write(self, text):
            if text:
                manifest.append({"type": "text", "value": text})
                old_stdout.write(text)

        def flush(self):
            old_stdout.flush()

    def _tee_publish(data, metadata=None, **kwargs):
        nonlocal img_idx
        png = (data or {}).get("image/png")
        if png:
            img_path = f"{base}_img{img_idx}.png"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(png))
            manifest.append({"type": "image", "path": img_path})
            img_idx += 1
        old_publish(data, metadata=metadata, **kwargs)  # still show it live

    sys.stdout = _TeeWriter()
    ip.display_pub.publish = _tee_publish  # type: ignore
    try:
        ip.run_cell(cell)
    finally:
        sys.stdout = old_stdout
        ip.display_pub.publish = old_publish

    after = set(ip.user_ns.keys())

    # Save only variables newly created by this cell, and only if picklable.
    to_save = {}
    skipped = []
    for name in after - before:
        val = ip.user_ns[name]
        try:
            pickle.dumps(val)
            to_save[name] = val
        except Exception:  # noqa: BLE001
            skipped.append(name)

    with open(vars_path, "wb") as f:
        pickle.dump(to_save, f)
    with open(manifest_path, "wb") as f:
        pickle.dump(manifest, f)

    print(
        f"[cache saved: '{label}' ({code_hash})]"
        + (f" — skipped unpicklable: {skipped}" if skipped else "")
    )


def load_ipython_extension(ipython):
    """Registers `%%cache` (called by `%load_ext`)."""
    ipython.register_magic_function(cache, magic_kind="cell")
