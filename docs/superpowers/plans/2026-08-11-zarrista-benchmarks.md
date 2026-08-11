# zarrista Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add zarrista to the `read_all`, `read_chunks`, and `read_subchunks` benchmarks, then re-run those three locally so all five implementations are measured on one machine.

**Architecture:** One new standalone CLI script honoring the existing shared contract (`path`, `--read_all`, `--concurrent_chunks N`, `--subchunks`), registered in three runners and the plotting module. zarrista uses its sync API on the native `FilesystemStore`; concurrency is N Python worker threads (it fully releases the GIL) rather than asyncio tasks, because its async API cannot accept a filesystem store.

**Tech Stack:** Python 3.13, uv, click, numpy, zarrista 0.1.0b8, zarrs-python 0.2.3, zarr-python 3.1.5, tensorstore 0.1.80, matplotlib, pandas, just, zarrs_tools 0.7.5 (Rust).

**Design spec:** [docs/superpowers/specs/2026-08-11-zarrista-benchmarks-design.md](../specs/2026-08-11-zarrista-benchmarks-design.md)

## Global Constraints

- Implementation key is exactly `zarrista_python`; script filename is exactly `src/zarr_benchmarks/zarrista_python_benchmark_read.py`.
- zarrista pinned to `zarrista==0.1.0b8`; zarrs-python pinned to `zarrs==0.2.3` (from PyPI, never a local path).
- All benchmark scripts invoked by runners must be executable (`chmod +x`) and start with `#!/usr/bin/env python3`.
- Only `purge` / `drop_caches` runs under `sudo`. Benchmark processes run as the normal user under `gtime -v`; running them as root would distort RSS measurements.
- A failed cache clear must raise and abort the run. Silently producing warm-cache numbers is the worst possible outcome.
- zarrista reads pass `concurrent_target=1` so the thread count is the sole parallelism axis. `--read_all` uses library defaults instead.
- `read_subchunks` shares ONE `shard_cache()` across all threads (verified thread-safe).
- Do not modify the existing per-implementation benchmark scripts. Changes are confined to the new script, the runners' registration blocks, the plotting module, `__init__.py`, `pyproject.toml`, and `README.md`.
- Benchmarks run from the repo root (paths like `data/benchmark.zarr` and `./src/...` are relative to it).

**Verified zarrista API facts** (checked against `data/benchmark_compress_shard.zarr`; do not re-derive):

| Member | Kind | Value on the sharded dataset |
|---|---|---|
| `array.shape` | property | `[1024, 1024, 1024]` |
| `array.is_sharded` | property | `True` |
| `array.chunk_grid_shape` | property | `[4, 4, 4]` (shards) |
| `array.subchunk_grid_shape` | property | `[16, 16, 16]` |
| `array.subchunk_shape` | property | `[64, 64, 64]` |
| `array.subset_all` | property | `tuple` of slices |
| `array.shard_cache()` | **method** | returns `ShardCache` |
| `array[...]` | — | returns `Tensor`, `.to_numpy()` → `(1024,1024,1024) uint16` |

`retrieve_chunk(idx, concurrent_target=1)` and `retrieve_subchunk(idx, shard_cache=cache, concurrent_target=1)` both work; chunk indices are `list[int]`.

---

### Task 1: Make the project installable

The repo cannot currently run at all: `pyproject.toml` pins zarrs to a path on the original author's machine.

**Files:**
- Modify: `pyproject.toml:21` (the `zarrs` dependency) and the dependency list

- [ ] **Step 1: Confirm the failure**

Run: `uv run python -c "import zarr"`
Expected: FAIL with `error: Distribution not found at: file:///home/ljd/dev/zarrs-python`

- [ ] **Step 2: Repoint zarrs and add zarrista**

In `pyproject.toml`, replace this line:

```toml
    "zarrs @ file:///home/ljd/dev/zarrs-python",
```

with:

```toml
    "zarrs==0.2.3",
    "zarrista==0.1.0b8",
```

(An exact `==` pin naming a pre-release is accepted by uv without a `--prerelease` flag, per PEP 440.)

- [ ] **Step 3: Install gnu-time**

`time_args()` returns `["gtime", "-v"]` on Darwin and the runners will fail immediately without it.

```bash
brew install gnu-time
command -v gtime
```

Expected: a path is printed.

- [ ] **Step 4: Verify the environment resolves**

```bash
uv run python -c "import zarr, zarrs, zarrista, tensorstore, dask; print(zarr.__version__, zarrista.__version__)"
```

Expected: PASS, printing `3.1.5 0.1.0-beta.8`.

- [ ] **Step 5: Verify the Rust binaries and data are present**

```bash
zarrs_benchmark_read_sync --version && zarrs_reencode --version && du -sh data/*
```

Expected: versions print, and three datasets are listed (~2.0G, ~83M, ~440M). If `zarrs_benchmark_read_sync` is missing, run `just setup` — a plain `cargo install zarrs_tools` will NOT build it, as it is gated behind `required-features = ["benchmark"]`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "fix: depend on published zarrs, add zarrista

The zarrs dependency pointed at a local path on the original author's
machine, so the project could not be installed anywhere else."
```

---

### Task 2: Fix cache clearing and add a sudo keep-alive

`clear_cache()` on Darwin runs `subprocess.call(['sync', '&&', 'sudo', 'purge'])`, which passes `&&`, `sudo`, and `purge` as *arguments to `sync`* with no shell to interpret them — it syncs and never purges. Neither platform branch checks the return code. This repo has never been run on macOS, so no published numbers are affected, but this run is the first to exercise the code.

**Files:**
- Modify: `src/zarr_benchmarks/__init__.py` (entire file)

**Interfaces:**
- Produces: `clear_cache() -> None` (raises `RuntimeError` on failure), `time_args() -> list[str]`, `ensure_sudo() -> None`

- [ ] **Step 1: Write the failing test**

Create `/tmp/test_clear_cache.py`:

```python
"""Verify clear_cache raises instead of silently leaving the cache warm."""
import subprocess
import sys

from zarr_benchmarks import clear_cache

# Without cached sudo credentials, `sudo -n` cannot authenticate, so
# clear_cache MUST raise rather than return quietly.
subprocess.run(["sudo", "-k"], check=True)
try:
    clear_cache()
except RuntimeError as e:
    print("PASS: raised as expected:", e)
    sys.exit(0)
print("FAIL: clear_cache returned without clearing the cache")
sys.exit(1)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run python /tmp/test_clear_cache.py`
Expected: FAIL — prints `FAIL: clear_cache returned without clearing the cache`, exit code 1. (The current implementation calls `sync` with stray arguments and ignores the result.)

- [ ] **Step 3: Rewrite `src/zarr_benchmarks/__init__.py`**

```python
import atexit
import platform
import subprocess
import threading


def _clear_cache_command() -> list[str]:
    """The command that drops the OS page cache, per platform.

    `sh -c` is required: the exit status is the last command's, so a failed
    purge/drop_caches is visible in the return code.
    """
    system = platform.system()
    if system == "Darwin":
        return ["sudo", "-n", "sh", "-c", "sync; purge"]
    elif system == "Linux":
        return ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
    else:
        raise Exception("Unsupported platform")


def ensure_sudo():
    """Authenticate once, then keep the sudo timestamp fresh for the whole run.

    Only cache clearing needs root. Benchmark processes run as the normal user
    so that their measured RSS is not distorted.
    """
    if subprocess.call(["sudo", "-v"]) != 0:
        raise RuntimeError(
            "sudo authentication failed. Cache clearing requires root, and "
            "measurements taken on a warm cache would be invalid."
        )

    stop = threading.Event()

    def refresh():
        while not stop.wait(60):
            subprocess.call(
                ["sudo", "-n", "-v"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    threading.Thread(target=refresh, daemon=True).start()
    atexit.register(stop.set)


def clear_cache():
    """Drop the OS page cache, or raise if it could not be dropped."""
    result = subprocess.run(_clear_cache_command(), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to clear the page cache (exit {result.returncode}). "
            f"Measurements would be warm-cache and invalid. "
            f"stderr: {result.stderr.strip()}"
        )


def time_args():
    system = platform.system()
    if system == "Darwin":
        return ["gtime", "-v"]
    elif system == "Linux":
        return ["time", "-v"]
    else:
        raise Exception("Unsupported platform")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python /tmp/test_clear_cache.py`
Expected: PASS — prints `PASS: raised as expected: Failed to clear the page cache (exit 1)...`, exit code 0.

- [ ] **Step 5: Verify the success path actually purges**

```bash
uv run python -c "
from zarr_benchmarks import ensure_sudo, clear_cache
ensure_sudo()
clear_cache()
print('cache cleared successfully')
"
```

Expected: one password prompt, then `cache cleared successfully`. This is the first confirmation that cold-cache measurement works on this machine at all.

- [ ] **Step 6: Commit**

```bash
git add src/zarr_benchmarks/__init__.py
git commit -m "fix: actually clear the page cache on macOS and fail loudly

The Darwin branch passed '&&', 'sudo', 'purge' as arguments to sync with
no shell, so it synced and never purged. Neither branch checked the
return code, making a failed clear invisible. Adds a one-prompt sudo
keep-alive so a long unattended run needs a single authentication."
```

---

### Task 3: The zarrista benchmark script

**Files:**
- Create: `src/zarr_benchmarks/zarrista_python_benchmark_read.py`
- Test: `/tmp/test_zarrista_read.py` (verification only; not committed, to keep the upstream PR diff minimal)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: an executable CLI accepting `PATH`, `--read_all`, `--concurrent_chunks INT`, `--subchunks`; prints `Decoded in <float>ms` on stdout. Runners rely only on the process exiting 0 and on `gtime -v`'s stderr, not on stdout format.

- [ ] **Step 1: Write the failing test**

Create `/tmp/test_zarrista_read.py`. It checks that all three modes read the *same data* zarr-python reads — a script that silently returned fill values would otherwise look fast and mean nothing.

```python
"""Verify the zarrista script's three read modes against zarr-python."""
import subprocess
import sys

import numpy as np
import zarr
from zarr.storage import LocalStore
from zarrista import Array
from zarrista.store import FilesystemStore

PATH = "data/benchmark_compress_shard.zarr"
SCRIPT = "./src/zarr_benchmarks/zarrista_python_benchmark_read.py"

truth = zarr.open(store=LocalStore(PATH, read_only=True), mode="r")
array = Array.open(FilesystemStore(PATH), path="/").read_only()

failures = []

# 1. Whole-array read matches zarr-python for a corner region.
region = array[0:128, 0:128, 0:128].to_numpy()
if not np.array_equal(region, truth[0:128, 0:128, 0:128]):
    failures.append("array subset mismatch")

# 2. Chunk (shard) reads land at the right offsets.
shard = array.chunk_shape([0, 0, 0])
for idx in ([0, 0, 0], [1, 2, 3], [3, 3, 3]):
    got = array.retrieve_chunk(idx, concurrent_target=1).to_numpy()
    sl = tuple(slice(i * s, (i + 1) * s) for i, s in zip(idx, shard))
    if not np.array_equal(got, truth[sl]):
        failures.append(f"chunk {idx} mismatch")

# 3. Subchunk reads through a shared cache land at the right offsets.
cache = array.shard_cache()
sub = array.subchunk_shape
for idx in ([0, 0, 0], [5, 9, 2], [15, 15, 15]):
    got = array.retrieve_subchunk(idx, shard_cache=cache, concurrent_target=1).to_numpy()
    sl = tuple(slice(i * s, (i + 1) * s) for i, s in zip(idx, sub))
    if not np.array_equal(got, truth[sl]):
        failures.append(f"subchunk {idx} mismatch")

# 4. The CLI itself exits 0 in every mode the runners use.
for args in (["--read_all"], ["--concurrent_chunks", "4"],
             ["--subchunks", "--concurrent_chunks", "4"]):
    proc = subprocess.run([SCRIPT, *args, PATH], capture_output=True, text=True)
    if proc.returncode != 0:
        failures.append(f"CLI {args} exited {proc.returncode}: {proc.stderr[-300:]}")
    elif "Decoded in" not in proc.stdout:
        failures.append(f"CLI {args} printed no timing")

print("FAILURES:", failures) if failures else print("PASS: all modes correct")
sys.exit(1 if failures else 0)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run python /tmp/test_zarrista_read.py`
Expected: FAIL — the CLI checks report a non-zero exit because the script does not exist yet.

- [ ] **Step 3: Write the script**

Create `src/zarr_benchmarks/zarrista_python_benchmark_read.py`:

```python
#!/usr/bin/env python3

import os
import timeit
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np

from zarrista import Array
from zarrista.store import FilesystemStore


@click.command()
@click.argument('path', type=str)
@click.option('--concurrent_chunks', type=int, default=None, help='Number of concurrent chunk reads. Ignored if --read-all is set')
@click.option('--subchunks', is_flag=True, show_default=True, default=False, help='Read inner-chunk-by-inner-chunk. Ignored if --read-all is set')
@click.option('--read_all', is_flag=True, show_default=True, default=False, help='Read the entire array in one operation.')
def main(path, concurrent_chunks, subchunks, read_all):
    # zarrista's async API accepts only obstore/icechunk/AsyncZipStore stores, so
    # a local-filesystem benchmark uses the sync API on the native store. zarrista
    # releases the GIL, so Python worker threads give real parallelism.
    array = Array.open(FilesystemStore(path), path="/").read_only()

    if subchunks:
        grid_shape = array.subchunk_grid_shape
        # One cache shared across threads (ShardCache is thread-safe): each
        # shard index is read once rather than once per subchunk.
        shard_cache = array.shard_cache()

        def read_one(chunk_index):
            return array.retrieve_subchunk(
                chunk_index, shard_cache=shard_cache, concurrent_target=1
            )
    else:
        grid_shape = array.chunk_grid_shape

        def read_one(chunk_index):
            return array.retrieve_chunk(chunk_index, concurrent_target=1)

    print("Domain shape", array.shape)
    print("Number of chunks", grid_shape)

    chunk_indices = [list(index) for index in np.ndindex(*grid_shape)]

    start_time = timeit.default_timer()
    if read_all:
        # Library defaults here: a single whole-array read, matching how the
        # other implementations issue their read-all.
        print(array[...].to_numpy().shape)
    else:
        workers = concurrent_chunks or os.cpu_count()
        if workers == 1:
            for chunk_index in chunk_indices:
                read_one(chunk_index)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # list() forces the generator so exceptions surface here.
                list(executor.map(read_one, chunk_indices))

    elapsed = timeit.default_timer() - start_time
    elapsed_ms = elapsed * 1000.0
    print(f"Decoded in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Make it executable**

```bash
chmod +x src/zarr_benchmarks/zarrista_python_benchmark_read.py
ls -l src/zarr_benchmarks/zarrista_python_benchmark_read.py
```

Expected: mode shows `-rwxr-xr-x`. The runners invoke it as `./src/...`, so this is required.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run python /tmp/test_zarrista_read.py`
Expected: PASS — prints `PASS: all modes correct`, exit code 0.

- [ ] **Step 6: Sanity-check the concurrency axis responds**

```bash
for n in 1 2 4 8; do
  printf "workers=%s " $n
  uv run ./src/zarr_benchmarks/zarrista_python_benchmark_read.py \
    --concurrent_chunks $n data/benchmark_compress.zarr | grep Decoded
done
```

Expected: times decrease as workers increase. A flat line means `concurrent_target=1` is not taking effect and the sweep would measure nothing — stop and investigate before running the benchmarks.

- [ ] **Step 7: Commit**

```bash
git add src/zarr_benchmarks/zarrista_python_benchmark_read.py
git commit -m "feat: add zarrista read benchmark script

Uses the sync API on the native FilesystemStore; concurrency is N Python
worker threads with concurrent_target=1, since zarrista's async API
cannot accept a filesystem store. read_subchunks shares one thread-safe
shard cache."
```

---

### Task 4: Register zarrista in the three read runners

Also drops the dask variants from the run, so 475 runs fit in roughly an hour.

**Files:**
- Modify: `src/zarr_benchmarks/run_benchmark_read_all.py:12-47`
- Modify: `src/zarr_benchmarks/run_benchmark_read_chunks.py:12-50`
- Modify: `src/zarr_benchmarks/run_benchmark_read_subchunks.py:12-56`

**Interfaces:**
- Consumes: the executable script from Task 3; `ensure_sudo` from Task 2.
- Produces: CSV/Markdown in `measurements/` whose second column level includes `zarrista_python` and excludes `zarr_dask_python` / `zarrs_dask_python`.

- [ ] **Step 1: Add the dict entry in `run_benchmark_read_all.py`**

After the `"zarrs_dask_python"` entry inside `implementation_to_args` (ends line 35), add:

```python
        "zarrista_python": [
            "./src/zarr_benchmarks/zarrista_python_benchmark_read.py",
            "--read_all",
        ],
```

- [ ] **Step 2: Update the `implementations` list in `run_benchmark_read_all.py`**

Replace the list at lines 38-47 with:

```python
    implementations = [
        "zarrs_rust",
        # "zarrs_rust_async_as_sync",
        # "zarrs_rust_async",
        "tensorstore_python",
        "zarr_python",
        "zarrs_python",
        "zarrista_python",
        # Dask variants excluded from this run; see the design spec.
        # "zarr_dask_python",
        # "zarrs_dask_python",
    ]
```

- [ ] **Step 3: Add `ensure_sudo()` to `run_benchmark_read_all.py`**

Change the import on line 8 to:

```python
from zarr_benchmarks import clear_cache, time_args, ensure_sudo
```

and make the first statement inside `main()` (immediately after `def main():`):

```python
    ensure_sudo()
```

- [ ] **Step 4: Apply the same three changes to `run_benchmark_read_chunks.py`**

Dict entry (after the `"zarrs_dask_python"` entry, which ends line 38):

```python
        "zarrista_python": [
            "./src/zarr_benchmarks/zarrista_python_benchmark_read.py",
            "--concurrent_chunks",
        ],
```

`implementations` list (replacing lines 41-50):

```python
    implementations = [
        "zarrs_rust",
        # "zarrs_rust_async_as_sync",
        # "zarrs_rust_async",
        "tensorstore_python",
        "zarr_python",
        "zarrs_python",
        "zarrista_python",
        # Dask variants excluded from this run; see the design spec.
        # "zarr_dask_python",
        # "zarrs_dask_python",
    ]
```

Plus the same import change and `ensure_sudo()` call as Steps 3.

- [ ] **Step 5: Apply the same changes to `run_benchmark_read_subchunks.py`**

Dict entry (after the `"zarrs_python"` entry, which ends line 43):

```python
        "zarrista_python": [
            "./src/zarr_benchmarks/zarrista_python_benchmark_read.py",
            "--subchunks",
            "--concurrent_chunks",
        ],
```

`implementations` list (replacing lines 47-56):

```python
    implementations = [
        "zarrs_rust",
        # "zarrs_rust_async_as_sync",
        # "zarrs_rust_async",
        "tensorstore_python",
        "zarr_python",
        "zarrs_python",
        "zarrista_python",
        # "zarr_dask_python",
        # "zarrs_dask_python",
    ]
```

Plus the same import change and `ensure_sudo()` call. Note this runner's dict has no dask entries and only one image (`data/benchmark_compress_shard.zarr`) — leave both as they are.

- [ ] **Step 6: Verify the wiring without running a benchmark**

```bash
uv run python -c "
import ast, pathlib
for f in ['read_all', 'read_chunks', 'read_subchunks']:
    src = pathlib.Path(f'src/zarr_benchmarks/run_benchmark_{f}.py').read_text()
    assert 'zarrista_python' in src, f
    assert 'ensure_sudo()' in src, f
    ast.parse(src)
    print(f, 'ok')
"
```

Expected: three `ok` lines.

- [ ] **Step 7: Commit**

```bash
git add src/zarr_benchmarks/run_benchmark_read_all.py \
        src/zarr_benchmarks/run_benchmark_read_chunks.py \
        src/zarr_benchmarks/run_benchmark_read_subchunks.py
git commit -m "feat: run zarrista in the three read benchmarks

Excludes the dask variants from this run to keep it near an hour, and
authenticates sudo once up front so the run is unattended."
```

---

### Task 5: Register zarrista in the plotting module

**Files:**
- Modify: `src/zarr_benchmarks/plot_benchmarks.py:12-15` (imports), `:44-51` (`IMPLEMENTATIONS`), `:59-68` (`COLORS`), `:412-421` (`main`)

**Interfaces:**
- Consumes: the `zarrista_python` column key produced by Task 4.
- Produces: `plots/benchmark_read_all.svg`, `plots/benchmark_read_chunks.svg`, `plots/benchmark_read_subchunks.svg`.

- [ ] **Step 1: Add the import**

After line 15 (`import tensorstore`), add:

```python
import zarrista
```

- [ ] **Step 2: Add the legend entry**

Inside the `IMPLEMENTATIONS` dict, after the `"zarrs_python"` entry, add:

```python
    "zarrista_python": f"developmentseed/zarrista ({zarrista.__version__})",
```

`zarrista.__version__` reports `0.1.0-beta.8` natively, so no `importlib.metadata` fallback is needed (unlike `zarrs` and `tensorstore` above it).

- [ ] **Step 3: Add the colour**

Inside the `COLORS` dict, after the `"zarrs_python"` entry, add:

```python
    "zarrista_python": _tab10[6],
}
```

(replacing the existing closing brace — `_tab10[6]` is unused by the other six implementations).

- [ ] **Step 4: Restrict `main()` to the three regenerated plots**

Replace the body of `main()` (lines 413-420) with:

```python
def main():
    # Only the three standalone read benchmarks are re-run locally. The dask and
    # roundtrip plots keep their committed data from the reference system; see
    # docs/superpowers/specs/2026-08-11-zarrista-benchmarks-design.md.
    plot_read_all(plot_dask=False, ymax=YMAX_READ_ALL)
    plot_read_chunks(plot_dask=False)
    plot_read_subchunks(plot_dask=False)
    # plt.show()
```

Calling the `plot_dask=True` variants would now select zero columns and fail, because the re-run CSVs contain no dask columns.

- [ ] **Step 5: Verify the module imports and the colour key resolves**

```bash
uv run python -c "
from zarr_benchmarks.plot_benchmarks import IMPLEMENTATIONS, COLORS
assert 'zarrista_python' in IMPLEMENTATIONS and 'zarrista_python' in COLORS
assert len(set(COLORS.values())) == len(COLORS), 'duplicate colour'
print(IMPLEMENTATIONS['zarrista_python'])
"
```

Expected: `developmentseed/zarrista (0.1.0-beta.8)`, with no duplicate-colour assertion.

- [ ] **Step 6: Commit**

```bash
git add src/zarr_benchmarks/plot_benchmarks.py
git commit -m "feat: plot zarrista alongside the other implementations"
```

---

### Task 6: Calibrate, then run read_all

The full matrix is 475 runs against only ~9 minutes of actual benchmarking work — the rest is per-run `purge` plus interpreter startup. Measure that overhead before committing to the long run.

**Files:**
- Modify (conditionally): `best_of` in the three runners
- Produces: `measurements/benchmark_read_all.csv`, `measurements/benchmark_read_all.md`

- [ ] **Step 1: Measure the per-run overhead**

```bash
uv run python -c "
import time, subprocess
from zarr_benchmarks import ensure_sudo, clear_cache, time_args
ensure_sudo()
t0 = time.time(); clear_cache(); purge = time.time() - t0
t0 = time.time()
subprocess.run(time_args() + ['./src/zarr_benchmarks/zarrista_python_benchmark_read.py',
                              '--read_all', 'data/benchmark_compress.zarr'],
               capture_output=True)
run = time.time() - t0
print(f'purge {purge:.1f}s + run {run:.1f}s = {purge+run:.1f}s per measurement')
print(f'475 runs -> {(purge+run)*475/60:.0f} min')
"
```

Expected: a projected total. Record it.

- [ ] **Step 2: Decide `best_of`**

If the projection exceeds ~60 minutes, set `best_of = 3` (from `5`) in all three runners — `run_benchmark_read_all.py:55`, `run_benchmark_read_chunks.py:59`, and the equivalent line in `run_benchmark_read_subchunks.py`. Results aggregate with `nanmin`, so best-of-3 is within noise of best-of-5. If the projection is comfortably under an hour, leave `best_of = 5` and note the measured figure in the commit message.

- [ ] **Step 3: Run read_all**

```bash
just benchmark_read_all
```

Expected: ~75 runs. This is the pipeline smoke test — the shortest benchmark, exercising cache clearing, `gtime` parsing, all five implementations, and CSV/Markdown writing.

- [ ] **Step 4: Verify no implementation silently failed**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('measurements/benchmark_read_all.csv', header=[0,1], index_col=0)
print(df)
assert 'zarrista_python' in df.columns.get_level_values(1), 'zarrista missing'
assert not df.isna().any().any(), f'NaNs present:\n{df.isna().sum()}'
print('OK: no NaNs, all five implementations present')
"
```

Expected: `OK: no NaNs, all five implementations present`. A NaN column means that implementation crashed or its output failed to parse on every repetition — investigate before running the longer benchmarks. NaNs arise silently because the runner falls back to `math.nan` whenever the return code is non-zero.

- [ ] **Step 5: Commit**

```bash
git add measurements/benchmark_read_all.csv measurements/benchmark_read_all.md
git commit -m "chore: read_all measurements including zarrista (M2 Pro)"
```

---

### Task 7: Run read_chunks and read_subchunks

**Files:**
- Produces: `measurements/benchmark_read_chunks.{csv,md}`, `measurements/benchmark_read_subchunks.{csv,md}`

- [ ] **Step 1: Run read_chunks**

```bash
just benchmark_read_chunks
```

Expected: 300 runs (12 rows × 5 implementations × 5 repetitions), the longest of the three.

- [ ] **Step 2: Verify it**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('measurements/benchmark_read_chunks.csv', header=[0,1], index_col=[0,1])
assert 'zarrista_python' in df.columns.get_level_values(1)
assert not df.isna().any().any(), f'NaNs:\n{df.isna().sum()}'
print(df['Time (s)']['zarrista_python'])
print('OK')
"
```

Expected: `OK`, with zarrista times decreasing as concurrency rises within each image.

- [ ] **Step 3: Run read_subchunks**

```bash
just benchmark_read_subchunks
```

Expected: 100 runs (4 rows × 5 implementations × 5 repetitions), on the sharded dataset only.

- [ ] **Step 4: Verify it**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('measurements/benchmark_read_subchunks.csv', header=[0,1], index_col=[0,1])
assert 'zarrista_python' in df.columns.get_level_values(1)
assert not df.isna().any().any(), f'NaNs:\n{df.isna().sum()}'
print(df['Time (s)'])
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add measurements/benchmark_read_chunks.csv measurements/benchmark_read_chunks.md \
        measurements/benchmark_read_subchunks.csv measurements/benchmark_read_subchunks.md
git commit -m "chore: read_chunks and read_subchunks measurements including zarrista (M2 Pro)"
```

---

### Task 8: Regenerate plots and update the README

**Files:**
- Modify: `src/zarr_benchmarks/plot_benchmarks.py:20` (`YMAX_READ_ALL`)
- Modify: `README.md:11-20` (implementations), `:45-50` (benchmark system), and the benchmark sections
- Produces: three regenerated SVGs in `plots/`

- [ ] **Step 1: Retune the y-axis limit**

`YMAX_READ_ALL = 1.2` is fitted to the reference hardware. Read the new maximum:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('measurements/benchmark_read_all.csv', header=[0,1], index_col=0)
print('max read_all time:', df['Time (s)'].max().max())
"
```

Set `YMAX_READ_ALL` to roughly 1.1x that value, rounded to one decimal. `plot_read_chunks` and `plot_read_subchunks` use log axes with no ymax, so they need no change.

- [ ] **Step 2: Regenerate the plots**

```bash
just plot
ls -l plots/benchmark_read_all.svg plots/benchmark_read_chunks.svg plots/benchmark_read_subchunks.svg
```

Expected: three files with fresh timestamps, no exception. The four dask/roundtrip SVGs keep their original timestamps.

- [ ] **Step 3: Visually confirm the plots**

Open the three SVGs. Check that zarrista appears in each legend with its version, that its bars/lines are the new colour, and that no bar is clipped without the fade effect.

- [ ] **Step 4: Update the README implementations list**

Under `## Implementations Benchmarked`, after the `zarr-developers/zarr-python` bullet block, add:

```markdown
- [`developmentseed/zarrista`](https://github.com/developmentseed/zarrista)
  - A low-level Zarr API for Python built on `zarrs`
  - Benchmarked via its synchronous API on a native `FilesystemStore`
```

- [ ] **Step 5: Update the benchmark system and add a methodology note**

Replace the `## Benchmark System` list with this machine, and add the caveats:

```markdown
## Benchmark System
- Apple MacBook Pro
  - CPU: Apple M2 Pro (10 cores)
  - Memory: 16GB
  - OS: macOS (Darwin 25.5.0)

> [!NOTE]
> The read benchmarks below were measured on the system above. The dask and
> round trip plots retain earlier measurements from a different machine
> (Intel Core Ultra 7 268V, 32GB, Arch Linux) and are not directly comparable
> to the read plots.

> [!NOTE]
> `zarrista` notes:
> - Its concurrency axis is Python worker threads, not asyncio tasks: its async
>   API cannot accept a filesystem store. Reads pass `concurrent_target=1` so
>   the thread count is the only parallelism axis.
> - The subchunk benchmark shares one `ShardCache` across threads, so each shard
>   index is read once. Other implementations are not given an equivalent cache.
```

- [ ] **Step 6: Verify the README renders and links resolve**

```bash
grep -n "zarrista\|M2 Pro\|ShardCache" README.md
```

Expected: the new implementation bullet, benchmark system, and both notes appear.

- [ ] **Step 7: Commit**

```bash
git add plots/ src/zarr_benchmarks/plot_benchmarks.py README.md
git commit -m "docs: regenerate read plots with zarrista, update benchmark system

Retunes YMAX_READ_ALL for the new hardware and documents zarrista's
thread-based concurrency axis and shard cache usage."
```

---

## Before opening the upstream PR

These are deliberate concessions for the local run, not changes upstream should take. Revert them in the PR branch:

1. **Re-enable the dask variants** in `run_benchmark_read_all.py` and `run_benchmark_read_chunks.py`, and restore the full `main()` in `plot_benchmarks.py`. Upstream re-runs everything on the reference system, where dask and roundtrip belong in the matrix.
2. **Restore `best_of = 5`** if Task 6 lowered it.
3. **Revert the `## Benchmark System` section** to the reference machine, keeping the zarrista methodology note.
4. **Keep** the `pyproject.toml` fix, `clear_cache()` fix, the zarrista script, and the runner/plot registrations — all four are genuine improvements.
