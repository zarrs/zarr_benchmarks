setup:
	cargo binstall -y zarrs_tools@0.7.5

binstall:
	curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

generate_data:
	./generate_benchmark_array.rs data/benchmark.zarr
	./generate_benchmark_array.rs --compress data/benchmark_compress.zarr
	./generate_benchmark_array.rs --compress --subchunk-shape 32,32,32 data/benchmark_compress_shard.zarr

benchmark_read_all:
	uv run run_benchmark_read_all

benchmark_read_chunks:
	uv run run_benchmark_read_chunks

benchmark_roundtrip:
	uv run run_benchmark_roundtrip

plot:
	uv run plot_benchmarks

benchmark_all: benchmark_read_all benchmark_read_chunks benchmark_roundtrip
