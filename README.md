# msgpack-numpy-rs

[![Crates.io](https://img.shields.io/crates/v/msgpack-numpy)](https://crates.io/crates/msgpack-numpy)
[![Docs.rs](https://docs.rs/msgpack-numpy/badge.svg)](https://docs.rs/msgpack-numpy)
[![License](https://img.shields.io/crates/l/msgpack-numpy)](#license)

This crate does what Python's [msgpack-numpy](https://github.com/lebedov/msgpack-numpy/) does in Rust, and a lot [faster](#benchmarks). It serializes and deserializes NumPy scalars and arrays to and from the [MessagePack](https://msgpack.org/) format, in the same deserialized formats as the Python counterpart, so they could interoperate with each other. It enables processing NumPy arrays from a different service in Rust through IPC, or saving Machine Learning results to disk (better paired with compression).

## Overview

* It supports `bool`, `u8`, `i8`, `u16`, `i16`, `f16` (through the [`half`](https://crates.io/crates/half) crate), `u32`, `i32`, `f32`, `u64`, `i64`, `f64`.
* No support for arrays with complex numbers (`'c'`), byte strings (`'S'`), unicode strings (`'U'`), or other non-primitive types as elements. No support for structured/tuple data types (`'V'`), or object-type data that need pickling (`'O'`) ([ref](https://github.com/lebedov/msgpack-numpy/blob/0.4.8/msgpack_numpy.py)).
* However, during deserialization, we allow unsupported types to be deserialized as the `Unsupported` variant. This ensures deserialization can continue and the supported portions of data can be used.
* Scalars and arrays are represented as separate types, each of which being an enum of different element type variants. They come with convenient conversion methods (backed by the [`num-traits`](https://crates.io/crates/num-traits) crate) to the desired target primitive types. Example: `f16`, `f32`, `f64` can all be converted to `f64`, or `f16` with loss. This allows flexibility during deserialization, without explicit pattern matching and conditional conversion. It would be similar to NumPy's `.astype(np.float64)` / `.astype(np.float16)`. Notably, `bool` is convertible to numeric types as `(0, 1)`, but not from numeric types using these methods. Of course, you can do your own conversion after matching with the `Bool` variant.
* Arrays use the [`ndarray`](https://crates.io/crates/ndarray) crate, and have dynamic shapes. This enables users to leverage Rust's numeric [ecosystem](https://docs.rs/ndarray/latest/ndarray/index.html#the-ndarray-ecosystem) for the deserialized arrays.
* Array handling using `CowNDArray` could be zero-copy when array buffers in the serialized slice have good alignment (although MessagePack doesn't guarantee this all the time).
* It depends on [`serde`](https://crates.io/crates/serde). In addition, it makes sense to use a correct MessagePack implementation, such as [`rmp-serde`](https://crates.io/crates/rmp-serde), which is used in the examples below, although it doesn't need to be a dependency, due to `serde`'s design.

## Motivation

There hasn't been consensus on a good format that is both flexible and efficient for serializing NumPy arrays. They are unique in that they are blocks of bytes in nature, but also have numeric types and shapes. Programmers working on Machine Learning problems found MessagePack to have interesting properties. It is compact with a [type system](https://github.com/msgpack/msgpack/blob/master/spec.md), and has a wide range of language support. The package [msgpack-numpy](https://github.com/lebedov/msgpack-numpy/) provides de-/serialization for NumPy arrays, standalone or enclosed in arbitrary organizational depths, to be sent over the network, or saved to disk, in a compact format.

If one looks for a more production-oriented, performant format, they might consider [Apache Arrow](https://arrow.apache.org/), [Parquet](https://parquet.apache.org/), or [Protocol Buffers](https://protobuf.dev/). However, these formats are not as flexible as MessagePack when you need to store intermediate Machine Learning results. In practice, MessagePack with Numpy array support can be quite a good choice for many of these use cases.

This Rust version aims to provide a faster alternative to the Python version, with the same deserialized formats as the Python counterpart so they could interoperate with each other. You could use this as a building block for your own Machine Learning pipeline in Rust, or as a way to communicate between Python and Rust.

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

This applies to the owned `NDArray`.

| Array Type | Array Size | Arrays | Operation    | Python (ms) | Rust (ms) | Speedup |
|------------|-----------:|-------:|--------------|------------:|----------:|--------:|
|  f32       | 1000       | 10000  | Serialize    |       56.4  |     17.1  |    3.3x |
|            |            |        | Deserialize  |       26.1  |     18.9  |    1.4x |
|            | 100        | 100000 | Serialize    |      226.1  |     27.1  |    8.3x |
|            |            |        | Deserialize  |      199.3  |     50.5  |    3.9x |
|  f16       | 1000       | 10000  | Serialize    |       33.5  |      4.0  |    8.5x |
|            |            |        | Deserialize  |       21.2  |      5.2  |    4.1x |
|            | 100        | 100000 | Serialize    |      198.9  |     12.1  |   16.5x |
|            |            |        | Deserialize  |      195.2  |     29.5  |    6.6x |

The Rust implementation shows significant performance improvements over Python in all cases, with particularly dramatic speedups for small array serialization. The Python version's de-/serialization logic is written in C through NumPy, but small arrays reduce this benefit because each array is a Python object. Notably, the Python version deserializes faster than serializing, while the Rust version serializes faster than deserializing. This range of array sizes is typical for Machine Learning use cases, such as feature embeddings, so Rust will be able to help out when performance is needed.

### Zero-Copy Deserialization (when Good Alignment)

For the above arrays, the array buffers always seem to be misaligned during deserialization, so we can't just borrow the data from the serialized slice as the targeted typed array, but instead pay for extra allocation. This is because the MessagePack format doesn't guarantee alignment.

In most cases however, there are good chances of alignment, and we could borrow the array buffer data directly when that happens. This is demonstrated in the following benchmarks. We choose `CowNDArray`, shape `(1024, 2048)`, 10 arrays each time for demonstration.

| Data Type | Operation                 | Python (ms) | Rust (ms) | Speedup |
|-----------|---------------------------|------------:|----------:|--------:|
| f16       | Serialize                 |       42.8  |     23.4  |    1.8x |
|           | Deserialize (`NDArray`)   |       21.6  |     20.4  |    1.1x |
|           | Deserialize (`CowNDArray`)|       -     |     10.5  |    2.1x |
| f32       | Serialize                 |       87.8  |     43.5  |    2.0x |
|           | Deserialize (`NDArray`)   |       44.2  |     41.4  |    1.1x |
|           | Deserialize (`CowNDArray`)|       -     |     34.5  |    1.3x |

Deserialization time went down! For `f16`, it's about half the chance for good alignment, and 1/4 for `f32`. The amortized cost of allocation is now lower, and we can see the benefit of zero-copy deserialization. The shortcoming is, `CowNDArray` only supports `rmp_serde::from_slice` (consuming from a slice that's fully in memory), but not `rmp_serde::from_read` (consuming from a reader in a streaming way). So you need to keep the serialized bytes (the compiler will check).

If you really want complete zero-copy deserialization, you should try some other format, like [Apache Arrow](https://arrow.apache.org/).

## Notes

There is not a good reason to serialize using `Scalar`, because you end up representing primitive types with a lot of metadata. This type exists for compatibility reasons - it helps deserialize scalars already serialized this way.

## License
This project is licensed under the MIT license.
