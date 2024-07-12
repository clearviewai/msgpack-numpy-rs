use half::f16;
use msgpack_numpy::NDArray;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

fn bench_serialize<T: Serialize + for<'de> Deserialize<'de>>(
    value: &T,
    iterations: u32,
) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        rmp_serde::to_vec_named(&value).unwrap();
    }
    let total_duration = start.elapsed();
    total_duration / iterations
}

fn bench_deserialize<T: Serialize + for<'de> Deserialize<'de>>(
    value: &T,
    iterations: u32,
) -> Duration {
    let buf = rmp_serde::to_vec_named(&value).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _deserialized: Vec<NDArray> = rmp_serde::from_slice(&buf).unwrap();
    }
    let total_duration = start.elapsed();
    total_duration / iterations
}

fn main() {
    let iterations = 10; // Adjust this number as needed

    // large arrays
    let arr = NDArray::F16(
        Array1::from_iter(0..1000)
            .into_dyn()
            .mapv(|v| f16::from_f32(v as f32)),
    );
    let arrs = vec![arr; 10000];

    // small but many arrays
    // let arr = NDArray::F16(Array1::from_iter(0..100).into_dyn().mapv(|v| f16::from_f32(v as f32)));
    // let arrs = vec![arr; 100000];

    println!("Running benchmarks with {} iterations each...", iterations);

    let serialize_avg = bench_serialize(&arrs, iterations);
    println!("Serialize average: {:?}", serialize_avg);

    let deserialize_avg = bench_deserialize(&arrs, iterations);
    println!("Deserialize average: {:?}", deserialize_avg);
}
