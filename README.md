# msgpack-numpy-rs

[![Crates.io](https://img.shields.io/crates/v/msgpack-numpy)](https://crates.io/crates/msgpack-numpy)
[![Docs.rs](https://docs.rs/msgpack-numpy/badge.svg)](https://docs.rs/msgpack-numpy)
![License](https://img.shields.io/crates/l/msgpack-numpy)

This crate does what Python's [msgpack-numpy](https://github.com/lebedov/msgpack-numpy/) does in Rust. It serializes and deserializes NumPy scalars and arrays to and from [MessagePack](https://msgpack.org/) format, in the same deserialized formats as the Python version so they could interoperate with each other.

## Overview

* It supports `bool`, `u8`, `i8`, `u16`, `i16`, `f16` (through the [`half`](https://crates.io/crates/half) crate), `u32`, `i32`, `f32`, `u64`, `i64`, `f64`.
* No support for arrays with complex numbers (`'c'`), byte strings (`'S'`), unicode strings (`'U'`), or other non-primitive types as elements. No support for structured/tuple data types (`'V'`), or object-type data that need pickling (`'O'`) ([ref](https://github.com/lebedov/msgpack-numpy/blob/0.4.8/msgpack_numpy.py)).
* However, during deserialization, we allow unsupported types to be deserialized as the `Unsupported` variant. This ensures deserialization can continue and the supported portions of data can be used.
* Scalars and arrays are represented as separate types, each of which being an enum of different element type variants. They come with convenient conversion methods (backed by the [`num-traits`](https://crates.io/crates/num-traits) crate) to the desired target primitive types. Example: `f16`, `f32`, `f64` can all be converted to `f64`, or `f16` with loss. This allows flexibility during deserialization, without explicit pattern matching and conditional conversion. It would be similar to NumPy's `.astype(np.float64)` / `.astype(np.float16)`. Notably, `bool` is convertible to numeric types as `(0, 1)`, but not from numeric types using these methods. Of course, you can do your own conversion after matching with the `Bool` variant.
* Arrays use the [`ndarray`](https://crates.io/crates/ndarray) crate, and have dynamic shapes. This enables users to leverage Rust's numeric [ecosystem](https://docs.rs/ndarray/latest/ndarray/index.html#the-ndarray-ecosystem) for the deserialized arrays. Array handling is zero-copy if the types match - you only pay for conversion when you ask for it.
* It depends on [`serde`](https://crates.io/crates/serde). In addition, it makes sense to use a correct MessagePack implementation, such as [`rmp-serde`](https://crates.io/crates/rmp-serde), which is used in the examples below, although it doesn't need to be a dependency, due to `serde`'s design.

## Motivation

There hasn't been consensus on a good format that is both flexible and efficient for serializing NumPy arrays. They are unique in that they are blocks of bytes in nature, but also have numeric types and shapes. Programmers working on Machine Learning problems found MessagePack to have interesting properties. It is compact with a [type system](https://github.com/msgpack/msgpack/blob/master/spec.md), and has a wide range of language support. The package [msgpack-numpy](https://github.com/lebedov/msgpack-numpy/) provides de-/serialization for NumPy arrays, standalone or enclosed in arbitrary organizational depths, to be sent over the network, or saved to disk, in a compact format.

If one looks for a more production-oriented, performant format, they might consider [Apache Arrow](https://arrow.apache.org/), [Parquet](https://parquet.apache.org/), or [Protocol Buffers](https://protobuf.dev/). However, these formats are not as flexible as MessagePack when you need to store intermediate Machine Learning results. In practice, MessagePack with Numpy array support can be quite a good choice for many of these use cases.

This Rust version aims to provide a faster alternative to the Python version, with the same deserialized formats as the Python version so they could interoperate with each other. You could use this as a building block for your own Machine Learning pipeline in Rust, or as a way to communicate between Python and Rust.

## Examples

```rust
use std::fs::File;
use std::io::Read;
use msgpack_numpy::NDArray;

fn main() {
    let filepath = "tests/data/ndarray_bool.msgpack";
    let mut file = File::open(filepath).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    let deserialized: NDArray = rmp_serde::from_slice(&buf).unwrap();

    match &deserialized {
        NDArray::Bool(array) => {
            println!("{:?}", array);
        }
        _ => panic!("Expected NDArray::Bool"),
    }

    // returns an Option, None if conversion is not possible
    let arr = deserialized.into_u8_array().unwrap();
    println!("{:?}", arr);
}
```

Please see more in `examples/`.

## Benchmarks

All benchmarks were done with 1 CPU core on a Ubuntu 22.04 instance. CPUs: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz. The Rust version was compiled in release mode. We are only benchmarking the serialization and deserialization of arrays, in memory. See `benches/` for the benchmark code.

| Array Type | Array Size | Arrays | Operation    | Python (ms) | Rust (ms) | Speedup |
|------------|------------|-------:|--------------|------------:|----------:|--------:|
| `f32`      | 1000       | 10000  | Serialize    |     56.399  |   17.137  |   3.29x |
|            |            |        | Deserialize  |     26.106  |   18.940  |   1.38x |
|            | 100        | 100000 | Serialize    |    226.051  |   27.102  |   8.34x |
|            |            |        | Deserialize  |    199.253  |   50.509  |   3.94x |
| `f16`      | 1000       | 10000  | Serialize    |     33.534  |    3.953  |   8.48x |
|            |            |        | Deserialize  |     21.246  |    5.177  |   4.10x |
|            | 100        | 100000 | Serialize    |    198.863  |   12.058  |  16.49x |
|            |            |        | Deserialize  |    195.150  |   29.462  |   6.62x |

The Rust implementation shows significant performance improvements over Python in all cases, with particularly dramatic speedups for small array serialization. The Python version's de-/serialization logic is written in C through NumPy, but small arrays reduce this benefit because each array is a Python object. Notably, the Python version deserializes faster than serializing, while the Rust version serializes faster than deserializing. This range of array sizes is typical for Machine Learning use cases, such as feature embeddings, so Rust will be able to help out when performance is needed.

## License
This project is licensed under the MIT license.
