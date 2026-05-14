"""
Storage-format benchmark for lamina hidden-state records.

Each record contains:
  - input_hidden_states_mean:  float32 (33, 4096)
  - output_hidden_states_mean: float32 (33, 4096)
  - logits:                    float32 (1, 128256)
  - span_subject:              float32 (33, 4096)

Formats tested (skipped gracefully if package missing):
  npz_compressed  - np.savez_compressed  (current lamina format)
  npz_raw         - np.savez             (no compression)
  npy_dir         - one .npy file per array per record
  hdf5_gzip       - h5py + gzip level 4
  hdf5_lzf        - h5py + lzf
  safetensors     - HuggingFace safetensors (mmap-friendly, no compression)
  zarr_blosc      - zarr + blosc compressor

Metrics: median write time, median read time (5 trials each), total disk size (MB).
"""

import os
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_RECORDS = 100
N_TRIALS = 5
SEED = 42

ARRAY_SHAPES = {
    "input_hidden_states_mean": (33, 4096),
    "output_hidden_states_mean": (33, 4096),
    "logits": (1, 128256),
    "span_subject": (33, 4096),
}


# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------
def generate_records(n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        rec = {k: rng.random(shape, dtype=np.float32) for k, shape in ARRAY_SHAPES.items()}
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 ** 2)


def median_time(fn, n_trials: int = N_TRIALS) -> float:
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Format implementations
# ---------------------------------------------------------------------------

def bench_npz_compressed(records, tmpdir):
    def write():
        for i, rec in enumerate(records):
            path = os.path.join(tmpdir, f"{i:04d}.npz")
            np.savez_compressed(path, **rec)

    def read():
        for i in range(len(records)):
            path = os.path.join(tmpdir, f"{i:04d}.npz")
            with np.load(path) as f:
                _ = {k: f[k] for k in f.files}

    write()  # pre-populate for read benchmark
    w = median_time(write)
    r = median_time(read)
    size = dir_size_mb(tmpdir)
    return w, r, size


def bench_npz_raw(records, tmpdir):
    def write():
        for i, rec in enumerate(records):
            path = os.path.join(tmpdir, f"{i:04d}.npz")
            np.savez(path, **rec)

    def read():
        for i in range(len(records)):
            path = os.path.join(tmpdir, f"{i:04d}.npz")
            with np.load(path) as f:
                _ = {k: f[k] for k in f.files}

    write()
    w = median_time(write)
    r = median_time(read)
    size = dir_size_mb(tmpdir)
    return w, r, size


def bench_npy_dir(records, tmpdir):
    def write():
        for i, rec in enumerate(records):
            rec_dir = os.path.join(tmpdir, f"{i:04d}")
            os.makedirs(rec_dir, exist_ok=True)
            for k, arr in rec.items():
                np.save(os.path.join(rec_dir, f"{k}.npy"), arr)

    def read():
        for i in range(len(records)):
            rec_dir = os.path.join(tmpdir, f"{i:04d}")
            _ = {k: np.load(os.path.join(rec_dir, f"{k}.npy")) for k in ARRAY_SHAPES}

    write()
    w = median_time(write)
    r = median_time(read)
    size = dir_size_mb(tmpdir)
    return w, r, size


def bench_hdf5(records, tmpdir, compression, compression_opts=None):
    import h5py

    path = os.path.join(tmpdir, "data.h5")

    def write():
        with h5py.File(path, "w") as f:
            for i, rec in enumerate(records):
                grp = f.create_group(str(i))
                for k, arr in rec.items():
                    kwargs = {"compression": compression}
                    if compression_opts is not None:
                        kwargs["compression_opts"] = compression_opts
                    grp.create_dataset(k, data=arr, **kwargs)

    def read():
        with h5py.File(path, "r") as f:
            for i in range(len(records)):
                grp = f[str(i)]
                _ = {k: grp[k][:] for k in ARRAY_SHAPES}

    write()
    w = median_time(write)
    r = median_time(read)
    size = dir_size_mb(tmpdir)
    return w, r, size


def bench_safetensors(records, tmpdir):
    from safetensors.numpy import save_file, load_file

    def write():
        for i, rec in enumerate(records):
            path = os.path.join(tmpdir, f"{i:04d}.safetensors")
            save_file(rec, path)

    def read():
        for i in range(len(records)):
            path = os.path.join(tmpdir, f"{i:04d}.safetensors")
            _ = load_file(path)

    write()
    w = median_time(write)
    r = median_time(read)
    size = dir_size_mb(tmpdir)
    return w, r, size


def bench_zarr_blosc(records, tmpdir):
    import zarr
    import inspect

    store_path = os.path.join(tmpdir, "store.zarr")

    # zarr v2 uses 'compressor', zarr v3 uses 'compressors' (plural)
    sig_params = inspect.signature(zarr.Group.create_array).parameters
    if "compressors" in sig_params:
        # zarr v3 — try BloscCodec, fall back to default
        try:
            from zarr.codecs import BloscCodec
            extra = {"compressors": [BloscCodec()]}
        except Exception:
            extra = {}
    else:
        # zarr v2
        try:
            from numcodecs import Blosc
            extra = {"compressor": Blosc(cname="lz4", clevel=5)}
        except Exception:
            extra = {}

    def write():
        store = zarr.open(store_path, mode="w")
        for i, rec in enumerate(records):
            grp = store.require_group(str(i))
            for k, arr in rec.items():
                grp.create_array(k, data=arr, chunks=arr.shape, overwrite=True, **extra)

    def read():
        store = zarr.open(store_path, mode="r")
        for i in range(len(records)):
            grp = store[str(i)]
            _ = {k: grp[k][:] for k in ARRAY_SHAPES}

    write()
    w = median_time(write)
    r = median_time(read)
    size = dir_size_mb(tmpdir)
    return w, r, size


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
FORMATS = [
    ("npz_compressed", bench_npz_compressed, {}),
    ("npz_raw",        bench_npz_raw,        {}),
    ("npy_dir",        bench_npy_dir,        {}),
    ("hdf5_gzip",      bench_hdf5,           {"compression": "gzip", "compression_opts": 4}),
    ("hdf5_lzf",       bench_hdf5,           {"compression": "lzf"}),
    ("safetensors",    bench_safetensors,    {}),
    ("zarr_blosc",     bench_zarr_blosc,     {}),
]


def run_benchmarks(records):
    results = []
    for name, fn, kwargs in FORMATS:
        tmpdir = tempfile.mkdtemp(prefix=f"lamina_bench_{name}_")
        try:
            print(f"  Benchmarking {name} ...", flush=True)
            w, r, size = fn(records, tmpdir, **kwargs)
            results.append({"name": name, "write_s": w, "read_s": r, "size_mb": size})
        except ImportError as e:
            print(f"  SKIP {name}: {e}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return results


def print_table(results):
    results_sorted = sorted(results, key=lambda x: x["read_s"])

    col_name  = 20
    col_write = 12
    col_read  = 12
    col_size  = 12

    header = (
        f"{'Format':<{col_name}}"
        f"{'Write (s)':>{col_write}}"
        f"{'Read (s)':>{col_read}}"
        f"{'Size (MB)':>{col_size}}"
    )
    sep = "-" * (col_name + col_write + col_read + col_size)

    print()
    print("=" * len(sep))
    print("  lamina storage-format benchmark")
    print(f"  {N_RECORDS} records x {N_TRIALS} trials — median times")
    print("=" * len(sep))
    print(header)
    print(sep)
    for r in results_sorted:
        print(
            f"{r['name']:<{col_name}}"
            f"{r['write_s']:>{col_write}.3f}"
            f"{r['read_s']:>{col_read}.3f}"
            f"{r['size_mb']:>{col_size}.1f}"
        )
    print(sep)
    print()


def print_recommendation(results):
    if not results:
        print("No results to analyse.")
        return

    sorted_read  = sorted(results, key=lambda x: x["read_s"])
    sorted_write = sorted(results, key=lambda x: x["write_s"])
    sorted_size  = sorted(results, key=lambda x: x["size_mb"])

    fastest_read  = sorted_read[0]
    fastest_write = sorted_write[0]
    smallest      = sorted_size[0]

    # Current format for comparison
    current = next((r for r in results if r["name"] == "npz_compressed"), None)

    print("RECOMMENDATION")
    print("-" * 56)
    print(f"  Fastest read  : {fastest_read['name']}  ({fastest_read['read_s']:.3f} s)")
    print(f"  Fastest write : {fastest_write['name']}  ({fastest_write['write_s']:.3f} s)")
    print(f"  Smallest size : {smallest['name']}  ({smallest['size_mb']:.1f} MB)")
    print()

    if current:
        print(f"  Current format (npz_compressed):")
        print(f"    write={current['write_s']:.3f} s  read={current['read_s']:.3f} s  size={current['size_mb']:.1f} MB")
        print()

    # Simple heuristic: score = normalised_read + 0.5*normalised_write + 0.3*normalised_size
    max_r = max(r["read_s"]  for r in results)
    max_w = max(r["write_s"] for r in results)
    max_s = max(r["size_mb"] for r in results)

    def score(r):
        return (r["read_s"] / max_r) + 0.5 * (r["write_s"] / max_w) + 0.3 * (r["size_mb"] / max_s)

    best = min(results, key=score)
    print(f"  Overall best (read + 0.5*write + 0.3*size): {best['name']}")
    print()

    advice = {
        "npz_compressed":
            "Good default — high compression, slow write. Fine for write-once / read-rarely workloads.",
        "npz_raw":
            "Fast writes, large on disk. Good for scratch/temp storage when disk is cheap.",
        "npy_dir":
            "Simple, no extra deps, parallel-friendly. Slightly more files but very portable.",
        "hdf5_gzip":
            "Compact and portable, but slower I/O. Best when inter-operability (MATLAB, R) matters.",
        "hdf5_lzf":
            "Better I/O than gzip HDF5 with reasonable compression. A solid HDF5 choice.",
        "safetensors":
            "Fastest reads via mmap, minimal compression. Ideal for training loops that re-read data.",
        "zarr_blosc":
            "Parallel-chunk I/O, cloud-native. Best for large-scale distributed or streaming workloads.",
    }
    print(f"  Notes on {best['name']}:")
    print(f"    {advice.get(best['name'], '')}")
    print("-" * 56)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Generating {N_RECORDS} synthetic records (seed={SEED}) …")
    records = generate_records(N_RECORDS)

    nbytes = sum(arr.nbytes for rec in records for arr in rec.values())
    print(f"Raw data size: {nbytes / (1024**2):.1f} MB  ({len(records)} records)")
    print()
    print(f"Running benchmarks ({N_TRIALS} trials each) …")

    results = run_benchmarks(records)
    print_table(results)
    print_recommendation(results)
