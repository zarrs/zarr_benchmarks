#!/usr/bin/env -S cargo +nightly -Zscript

---cargo
[dependencies]
clap = { version = "4.5", features = ["derive"] }
indicatif = "0.18.3"
zarrs = { version = "0.22.10", features = ["zstd"] }
---

use clap::Parser;
use std::sync::Arc;
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;

#[derive(Clone, Debug)]
struct Shape([u64; 3]);

// 1. Allow it to be printed (fixes default_value_t error)
impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{},{}", self.0[0], self.0[1], self.0[2])
    }
}

// 2. Allow it to be parsed (fixes ValueParser error)
impl std::str::FromStr for Shape {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err("Expected 3 comma-separated numbers".to_string());
        }
        let res: Result<Vec<u64>, _> = parts.into_iter().map(|v| v.parse()).collect();
        match res {
            Ok(v) => Ok(Shape(v.try_into().unwrap())),
            Err(e) => Err(e.to_string()),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "generate_benchmark_array")]
#[command(about = "Generate benchmark Zarr arrays", long_about = None)]
struct Args {
    /// Output path for the Zarr array
    output_path: String,

    /// Array shape.
    #[arg(long, default_value_t = Shape([1024, 2048, 2048]))]
    array_shape: Shape,

    /// Chunk shape.
    #[arg(long, default_value_t = Shape([256, 256, 256]))]
    chunk_shape: Shape,
    /// Subchunk (inner-chunk) shape for sharded arrays.
    #[arg(long)]
    subchunk_shape: Option<Shape>,

    /// Compress the array
    #[arg(long, default_value_t = false)]
    compress: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let array_shape = args.array_shape.0;
    let chunk_shape = args.chunk_shape.0;
    let subchunk_shape = args.subchunk_shape.as_ref().map(|s| s.0);

    // Create the store
    let store = Arc::new(FilesystemStore::new(&args.output_path)?);

    // Build the array
    let mut array_builder = ArrayBuilder::new(
        array_shape.clone(),
        chunk_shape,
        DataType::UInt16,
        FillValue::from(0u16),
    );

    // Add compression if requested
    let zstd_codec = Arc::new(zarrs::array::codec::bytes_to_bytes::zstd::ZstdCodec::new(
        0.try_into().unwrap(),
        false,
    ));

    // Add sharding if requested
    if let Some(subchunk_shape) = subchunk_shape {
        use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
        let mut sharding_codec = ShardingCodecBuilder::new(subchunk_shape.try_into().unwrap());
        if args.compress {
            sharding_codec.bytes_to_bytes_codecs(vec![zstd_codec]);
        }
        array_builder.array_to_bytes_codec(Arc::new(sharding_codec.build()));
    } else if args.compress {
        array_builder.bytes_to_bytes_codecs(vec![zstd_codec]);
    }

    // Build and create the array
    let array = array_builder.build(store.clone(), "/")?;
    array.store_metadata()?;

    // Iterate over the array chunk-by-chunk and fill with benchmark data
    let num_chunks = array.chunk_grid().grid_shape();
    let bar = indicatif::ProgressBar::new(num_chunks.iter().product());

    let chunks = array.chunk_grid().iter_chunk_indices();
    chunks.for_each(|chunk_idx| {
        let chunk_subset = array.chunk_subset(&chunk_idx).unwrap();
        let chunk_offset = chunk_subset.start();
        let chunk_shape = chunk_subset.shape();
        let mut chunk_data = vec![0u16; chunk_shape.iter().product::<u64>() as usize];
        for l0 in 0..chunk_shape[0] {
            for l1 in 0..chunk_shape[1] {
                for l2 in 0..chunk_shape[2] {
                    let g0 = chunk_offset[0] + l0;
                    let g1 = chunk_offset[1] + l1;
                    let g2 = chunk_offset[2] + l2;

                    let value = ((g2 + (g1 * g1) / 32 + g0 * g0 * g0) % 65536) as u16;
                    let index = (l0 * chunk_shape[1] * chunk_shape[0]) + (l1 * chunk_shape[0]) + l2;
                    chunk_data[index as usize] = value;
                }
            }
        }
        array
            .store_array_subset_elements(&chunk_subset, &chunk_data)
            .unwrap();
        bar.inc(1);
    });
    bar.finish();

    println!("Array successfully written to {}", args.output_path);
    Ok(())
}
