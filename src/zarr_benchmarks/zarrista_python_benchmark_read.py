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

        # Each read returns nothing: the decoded data is dropped in the worker
        # thread. Retaining it would hold the whole array in memory and inflate
        # both the peak RSS and the timings. The other implementations discard
        # their chunks too (asyncio.TaskGroup drops each task as it completes).
        #
        # Codec options are left at their defaults so that N means the same
        # thing here as for the other implementations: N chunks in flight, each
        # free to use the library's internal parallelism. Pinning
        # concurrent_target=1 would make zarrista the only serial-at-N=1 line.
        def read_one(chunk_index):
            array.retrieve_subchunk(chunk_index, shard_cache=shard_cache)
    else:
        grid_shape = array.chunk_grid_shape

        def read_one(chunk_index):
            array.retrieve_chunk(chunk_index)

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
