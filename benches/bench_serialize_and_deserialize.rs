use half::f16;
use msgpack_numpy::NDArray;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

fn bench_serialize<T: Serialize>(value: &T, iterations: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        rmp_serde::to_vec_named(&value).unwrap();
    }
    let total_duration = start.elapsed();
    total_duration / iterations
}

fn bench_deserialize<'a, T: Serialize + Deserialize<'a>>(value: &T, iterations: u32) -> Duration {
    let buf = rmp_serde::to_vec_named(&value).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _deserialized: Vec<NDArray> = rmp_serde::from_slice(&buf).unwrap();
        // zero-copy deserialization if good alignment
        // use msgpack_numpy::CowNDArray;
        // let _deserialized: Vec<CowNDArray> = rmp_serde::from_slice(&buf).unwrap();
    }
    let total_duration = start.elapsed();
    total_duration / iterations
}

fn main() {
    let iterations = 10; // Adjust this number as needed

    // large arrays
    let arr = NDArray::F16(
        Array1::range(0., 1000., 1.)
            .into_dyn()
            .mapv(f16::from_f32)
            .into(),
    );
    let arrs = vec![arr; 10000];

    // small but many arrays
    // let arr = NDArray::F16(
    //    Array1::range(0., 1000., 1.)
    //         .into_dyn()
    //         .mapv(f16::from_f32)
    //         .into(),
    // );
    // let arrs = vec![arr; 100000];

    // test zero-copy deserialization
    // use ndarray::Array2;
    // let arr = NDArray::F16(
    //     Array2::zeros((1024, 2048))
    //         .into_dyn()
    //         .mapv(f16::from_f32)
    //         .into(),
    // );
    // let arrs = vec![arr; 10];

    println!("Running benchmarks with {} iterations each...", iterations);

    let serialize_avg = bench_serialize(&arrs, iterations);
    println!("Serialize average: {:?}", serialize_avg);

    let deserialize_avg = bench_deserialize(&arrs, iterations);
    println!("Deserialize average: {:?}", deserialize_avg);
}
