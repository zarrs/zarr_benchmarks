#!/usr/bin/env python3

import timeit
import click
import sys

import dask.array as da

import zarr

import zarrs
zarr.config.set({
    "threading.num_workers": None,
    "array.write_empty_chunks": False,
    "codec_pipeline": {
        'batch_size': 1,
        "path": "zarrs.ZarrsCodecPipeline",
        "validate_checksums": True,
        "store_empty_chunks": False,
        "chunk_concurrent_minimum": 4,
        "chunk_concurrent_maximum": None,
    }
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
