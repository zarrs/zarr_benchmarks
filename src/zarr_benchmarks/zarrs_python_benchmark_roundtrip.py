#!/usr/bin/env python3

import timeit
import click

import zarr
from zarr.storage import LocalStore, FsspecStore

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
    if path.startswith("http"):
        store = FsspecStore.from_url(url=path) # broken with zarr-python 3.0.0a0
    else:
        store = LocalStore(path, read_only=True)

    dataset = zarr.open(store=store, mode='r')
    dataset_out = zarr.create(store=LocalStore(output), mode='w', shape=dataset.shape, chunks=dataset.chunks, dtype=dataset.dtype, codecs=dataset.metadata.codecs)

    start_time = timeit.default_timer()

    dataset_out[:] = dataset[:]

    elapsed = timeit.default_timer() - start_time
    elapsed_ms = elapsed * 1000.0

    print(f"Round trip in {elapsed_ms:.2f}ms")

if __name__ == "__main__":
    main()
