#![doc = include_str!("../README.md")]

mod core;
mod serde;

pub use core::{CowNDArray, NDArray, Scalar};

// re-export ndarray
pub use ndarray;
