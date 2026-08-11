# Adding zarrista to the Zarr benchmarks

**Date:** 2026-08-11
**Status:** Approved, not yet implemented

## Goal

Add [zarrista](https://github.com/developmentseed/zarrista) — a low-level Python
Zarr API built on Rust `zarrs` — to this repository's read benchmarks, and
re-run the affected benchmarks locally so the numbers are directly comparable.

Two outcomes, in priority order:

1. **Results.** How does zarrista compare to `zarrs` (Rust), `tensorstore`,
   `zarr-python`, and `zarr-python` + `ZarrsCodecPipeline`? Only relative
   standings matter, so a single machine measuring all implementations under
   identical conditions is sufficient.
2. **Code.** The integration should eventually be upstreamable as a PR to
   `zarrs/zarr_benchmarks`, so it follows existing conventions and avoids
   gratuitous changes to other implementations.

## Background

Each benchmark runner shells out to a standalone CLI script per implementation,
wrapped in `gtime -v` (macOS) or `time -v` (Linux), parses wall time and maximum
resident set size from stderr, takes the best of N repetitions, and writes a CSV
plus a Markdown table to `measurements/`. `plot_benchmarks.py` renders SVGs from
the CSVs.

Adding an implementation therefore means: one new script honoring the shared CLI
contract, one entry in each runner's `implementation_to_args` dict and
`implementations` list, and one entry each in `IMPLEMENTATIONS` and `COLORS` in
the plotting module.

zarrista version at time of writing: **0.1.0b8** (`__version__` reports
`0.1.0-beta.8`).

## Scope

zarrista participates in three benchmarks:

| Benchmark | Included | Reason |
|---|---|---|
| `read_all` | Yes | |
| `read_chunks` | Yes | |
| `read_subchunks` | Yes | `retrieve_subchunk` is a differentiator |
| `roundtrip` | **No** | Avoids exercising a pre-0.1 library's write path |

## Design

### 1. The zarrista benchmark script

New file `src/zarr_benchmarks/zarrista_python_benchmark_read.py`, executable,
with a `#!/usr/bin/env python3` shebang. It resolves to the project venv because
the runners are themselves launched via `uv run`, which puts `.venv/bin` on
`PATH`.

It honors the same CLI contract as the existing read scripts, so no runner needs
special-casing:

- `path` — positional argument
- `--read_all` — read the whole array in one operation
- `--concurrent_chunks N` — number of chunks in flight
- `--subchunks` — iterate inner chunks rather than whole chunks

Opening uses the sync API on the native filesystem store:

```py
from zarrista import Array
from zarrista.store import FilesystemStore

array = Array.open(FilesystemStore(path), path="/").read_only()
```

API mapping per mode:

| Mode | Call | Grid |
|---|---|---|
| `--read_all` | `array[...].to_numpy()` | — |
| chunk-by-chunk | `array.retrieve_chunk(idx)` | `array.chunk_grid_shape` |
| `--subchunks` | `array.retrieve_subchunk(idx, shard_cache=cache)` | `array.subchunk_grid_shape` |

zarrista's outer chunk corresponds to zarr-python's *shard* when the array is
sharded, and its subchunk corresponds to zarr-python's *chunk*. This matches
what the existing scripts select with `dataset.shards or dataset.chunks` versus
`dataset.chunks`, so all implementations measure the same work.

Only the read loop is timed, matching the existing scripts: `timeit.default_timer()`
around the loop, then print the elapsed milliseconds. Store opening and metadata
inspection sit outside the timed region.

### 2. Concurrency semantics

Every other Python script implements `--concurrent_chunks N` as N asyncio tasks
bounded by a semaphore. zarrista cannot use that shape here: its async API
accepts only `obstore.ObjectStore`, `icechunk.Session`, or `AsyncZipStore` —
`FilesystemStore` is sync-only. Using the async API on local disk would
therefore benchmark obstore's IO stack rather than zarrista's native store.

Instead, concurrency is **N Python worker threads over the sync API**:

```py
with ThreadPoolExecutor(max_workers=n) as ex:
    list(ex.map(lambda i: array.retrieve_chunk(i, concurrent_target=1), indices))
```

This is sound because zarrista fully releases the GIL on sync reads. Measured on
this machine with `RAYON_NUM_THREADS=1`, so rayon could not be responsible for
the scaling:

| Threads | Time |
|---|---|
| 1 (serial) | 255 ms |
| 2 | 129 ms |
| 4 | 72 ms |
| 8 | 45 ms |

`concurrent_target=1` is passed deliberately. Under zarrista's defaults a single
`retrieve_chunk` already fans out across all cores via rayon, so the library is
saturated at N=1 and the 1/2/4/8 sweep would plot a flat line, measuring nothing.
Pinning it to 1 makes N the sole parallelism axis and the curve interpretable.

This is not perfectly symmetric with the other implementations — `zarrs_python`
carries `chunk_concurrent_minimum: 4`, and zarr-python does some internal fan-out
at N=1 as well. No configuration is exactly fair; this one is at least legible,
and the choice is documented in the README.

For `--read_all`, zarrista's defaults are left alone (full internal rayon
parallelism), matching how the other implementations issue a single whole-array
read.

### 3. Shard cache

`read_subchunks` uses a `shard_cache`, created once via `array.shard_cache()`,
so each shard's index is read once rather than once per subchunk. This is the
API's intended usage and a realistic way the array would be read.

No other implementation is given an equivalent (tensorstore's `cache_pool` is
commented out in its script), so this is an advantage — it is called out in the
README rather than hidden.

`ShardCache` thread-safety is unverified. If it is not thread-safe, use one cache
per worker thread, which retains most of the benefit. **This must be checked
before the benchmark run.**

### 4. Correctness verification

zarrista is well tested upstream, so this is a smoke test rather than a test
suite: on a small locally-generated array, confirm that all three modes return
data equal to what zarr-python returns for the same regions.

This runs outside the timed path and is not committed as part of the benchmark.
Its purpose is narrow — a script that silently reads fill values or the wrong
subchunk would produce excellent timings that mean nothing, and that failure
would not be detected until after the full run. The existing scripts do not
validate their reads either (`zarr_python_benchmark_read.py` discards every
chunk it fetches), which is tolerable for implementations already known-good.

### 5. Cold cache and `clear_cache()`

Cold-cache measurement is retained. It matters asymmetrically — from the
committed `zarrs_rust` read_all numbers:

| Dataset | On disk | Cold time | Dominated by |
|---|---|---|---|
| Uncompressed | 2.00 GB | 0.58 s | IO (~3.4 GB/s, disk speed) |
| Compressed | 83 MB | 0.26 s | decode; IO under 10% |
| Compressed + sharded | 439 MB | 0.42 s | decode; IO ~0.1 s |

The uncompressed dataset is the *slowest* despite doing no decode work at all;
that ordering exists only because of cold-cache IO, and warm-cache runs would
reorder the datasets entirely. Warm cache subtracts a similar IO constant from
every implementation, so rankings would survive, but implementations that
overlap IO with decode (tensorstore) would silently lose a real advantage.

**Three bugs must be fixed in `src/zarr_benchmarks/__init__.py`:**

1. `subprocess.call(['sync', '&&', 'sudo', 'purge'])` passes `&&`, `sudo`, and
   `purge` as *arguments to `sync`*, with no shell to interpret them. It syncs
   and never purges. Every macOS measurement to date was taken on a warm page
   cache. Fix to `sudo -n sh -c 'sync; purge'`, whose exit status is `purge`'s,
   since it is the last command in the shell.
2. The return code is never checked — `subprocess.call`'s result is discarded on
   both the Darwin and Linux branches. A failed clear is therefore invisible.
   `purge` does signal failure correctly (verified: exit code 1 with
   `Unable to purge disk buffers: Operation not permitted` on stderr), so
   checking `returncode` is sufficient.
3. A failed cache clear must abort the run loudly. Silently producing warm-cache
   numbers is the worst outcome.

`purge` requires root — verified, there is no unprivileged alternative on macOS.
Only `purge` runs under `sudo`, as its own short-lived subprocess; the benchmark
processes themselves run as the normal user under `gtime -v` (running them as
root would distort the RSS measurements). The runners call `sudo -v` once at
startup and refresh the credential timestamp from a background thread every 60
seconds, so a single password prompt covers the entire run without any permanent
system configuration change.

### 6. Run matrix

Dask variants and the roundtrip benchmark are excluded from the re-run to keep
it under an hour. Five implementations remain: `zarrs_rust`,
`tensorstore_python`, `zarr_python`, `zarrs_python`, `zarrista_python`.

| Benchmark | Runs | Measured work |
|---|---|---|
| read_all | 3 img × 5 impl × 5 reps = 75 | ~1 min |
| read_chunks | 12 rows × 5 impl × 5 reps = 300 | ~5 min |
| read_subchunks | 4 rows × 5 impl × 5 reps = 100 | ~3 min |
| **Total** | **475** | **~9 min** |

The benchmarking work is only ~9 minutes; the remaining time is fixed per-run
overhead — one `purge` plus interpreter startup, 475 times over. The run
duration is therefore governed by run *count*, not by work.

475 runs sits at the edge of an hour: ~41 min at 4 s overhead per run, ~72 min
at 8 s. **A calibration step runs first** — time a single purge plus one
benchmark invocation — so the schedule is set from a measured number.
`best_of 5 → 3` is held in reserve as the lever that closes the gap if
calibration comes in high; it cuts run count by 40% at negligible cost, since
results are aggregated with `nanmin`.

`concurrent_chunks_list` stays at `[1, 2, 4, 8]`.

### 7. Plotting

- `IMPLEMENTATIONS["zarrista_python"] = f"developmentseed/zarrista ({zarrista.__version__})"`
- `COLORS["zarrista_python"] = _tab10[6]`
- `import zarrista` alongside the existing version imports

The existing dask/non-dask column filter needs no change: `"zarrista_python"`
does not match `"dask"`, so it lands in the standalone plots automatically.

Because dask and roundtrip are not re-run, `main()` regenerates only the three
standalone read plots. The committed dask and roundtrip CSVs and SVGs are left
untouched on disk — deleting upstream's results would make for a hostile PR
diff, and upstream re-runs everything on the reference system anyway.

`YMAX_READ_ALL` and the other `YMAX_*` constants are fitted to the reference
hardware and are retuned after the run so bars are not clipped arbitrarily.

### 8. Environment and setup

Benchmark system for this run: **Apple M2 Pro, 10 cores, 16 GB RAM, macOS**
(reference system in the README is an 8-core Intel Arch Linux box with 32 GB).

Status of prerequisites:

| Item | Status |
|---|---|
| `zarrs_tools` via `just setup` | Done — all nine binaries present |
| Benchmark data generated | Done — 2.0 G / 83 M / 440 M |
| `gtime` (`brew install gnu-time`) | **Outstanding** — `time_args()` fails without it |
| `zarrs` dependency pin | **Outstanding** — fatal, see below |

`pyproject.toml` pins `zarrs @ file:///home/ljd/dev/zarrs-python`, a path that
exists only on the original author's machine. The project currently cannot be
installed anywhere else — `uv run python -c "import zarr"` fails with
`Distribution not found`. Repoint to `zarrs==0.2.3` (latest on PyPI).

Note: `cargo install zarrs_tools --locked` does *not* build the benchmark
binaries; they are gated behind `required-features = ["benchmark"]`. `just setup`
uses `cargo binstall`, which pulls prebuilt release binaries that include them.

### 9. README updates

- Add zarrista to "Implementations Benchmarked"
- Update "Benchmark System" to the M2 Pro, noting that the dask and roundtrip
  figures still come from the reference system
- Document that zarrista's concurrency axis is Python worker threads with
  `concurrent_target=1`, not asyncio tasks
- Document that `read_subchunks` uses a shard cache for zarrista

## Execution order

1. Fix `pyproject.toml`; `brew install gnu-time`; confirm `uv run` works.
2. Fix `clear_cache()`; verify a purge succeeds and that failure aborts.
3. Write the zarrista script; run the correctness smoke test.
4. Wire zarrista into the three runners and the plotting module.
5. Calibrate: time one purge plus one invocation; decide on `best_of`.
6. Run `read_all` end-to-end as a pipeline smoke test; inspect output.
7. Run `read_chunks` and `read_subchunks`.
8. Retune `YMAX_*`, regenerate the three plots, update the README.

## Risks

- **16 GB RAM.** The read rows peaked around 2.5 GB on a 32 GB box, so they are
  safe. This is only a concern because dask and roundtrip — the rows that peaked
  near 4.4 GB — are excluded anyway.
- **`ShardCache` thread-safety** is unverified; see §3.
- **`concurrent_target=1` is a judgment call**, not a neutral default. It is
  documented in the README so readers are not left to guess.
- **Calibration may show the hour target is unreachable** at `best_of=5`, in
  which case drop to 3.
