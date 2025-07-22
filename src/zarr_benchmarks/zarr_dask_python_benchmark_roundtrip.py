#!/usr/bin/env python3

import timeit
import click
import sys
import zarr

import dask.array as da

zarr.config.set({
    "async.concurrency": None,
})

@click.command()
@click.argument('path', type=str)
@click.argument('output', type=str)
def main(path, output):
    # if "benchmark_compress_shard.zarr" in path:
    #     sys.exit(1)
    z = zarr.open_array(path)

    arr = da.from_zarr(path, chunks=z.shards)
    start_time = timeit.default_timer()
    da.to_zarr(arr, output)
    elapsed = timeit.default_timer() - start_time
    elapsed_ms = elapsed * 1000.0

    print(f"Round trip in {elapsed_ms:.2f}ms")

if __name__ == "__main__":
    main()
