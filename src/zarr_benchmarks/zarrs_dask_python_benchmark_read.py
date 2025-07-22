#!/usr/bin/env python3

import timeit
import click
import sys

import dask
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
@click.option('--concurrent_chunks', type=int, default=None, help='Number of concurrent async chunk reads. Ignored if --read-all is set')
@click.option('--read_all', is_flag=True, show_default=True, default=False, help='Read the entire array in one operation.')
def main(path, concurrent_chunks, read_all):
    # if "benchmark_compress_shard.zarr" in path:
    #     sys.exit(1)
    z = zarr.open_array(path)
    arr = da.from_zarr(path, chunks=z.shards)

    start_time = timeit.default_timer()
    if read_all:
        print(arr.compute().shape)
    else:
        if concurrent_chunks is not None:
            # _client = Client(threads_per_worker=concurrent_chunks, n_workers=1) # very high overhead
            # _client = Client(threads_per_worker=1, n_workers=concurrent_chunks) # very high overhead
            # dask.config.set(scheduler='processes', num_workers=concurrent_chunks) # very high overhead
            dask.config.set(scheduler='threads', num_workers=concurrent_chunks)

        @dask.delayed
        def read_chunk(chunk) -> None:
            # Do nothing with the chunk (just read it into memory)
            pass
        results = []
        for chunk in arr.to_delayed().ravel():
            results.append(read_chunk(chunk))
        dask.compute(results)

    elapsed = timeit.default_timer() - start_time
    elapsed_ms = elapsed * 1000.0
    print(f"Decoded in {elapsed_ms:.2f}ms")

if __name__ == "__main__":
    main()
