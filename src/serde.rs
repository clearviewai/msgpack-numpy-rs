use crate::core::{NDArray, Scalar};
use half::f16;
use serde::de::{self, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_bytes::{ByteBuf, Bytes};
use std::fmt;

// DType

enum DType {
    String(String),
    #[allow(dead_code)]
    Array(Vec<(String, String)>),
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DTypeVisitor;

        impl<'de> Visitor<'de> for DTypeVisitor {
            type Value = DType;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string or an array of tuples")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DType::String(value.to_string()))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut vec = Vec::new();
                while let Some((name, dtype)) = seq.next_element()? {
                    vec.push((name, dtype));
                }
                Ok(DType::Array(vec))
            }
        }

        deserializer.deserialize_any(DTypeVisitor)
    }
}

/***********************************************************************************************/
// Scalar

// impl Deserialize

impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ScalarVisitor;

        impl<'de> Visitor<'de> for ScalarVisitor {
            type Value = Scalar;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a numpy scaler in msgpack format")
            }

            // additional compatibility in case msgpack-python short-circuits during serialization
            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::Bool(v))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::I64(v))
            }

            // msgpack-python indeed short-circuits this during serialization
            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::F64(v))
            }

            // for NumPy's 'U' type
            fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::Unsupported)
            }

            // for NumPy's 'S' type
            fn visit_bytes<E>(self, _v: &[u8]) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::Unsupported)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut nd: Option<bool> = None;
                let mut numpy_dtype: Option<DType> = None;
                let mut data: Option<ByteBuf> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "nd" => nd = Some(map.next_value()?),
                        "type" => numpy_dtype = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => return Err(de::Error::unknown_field(key, &["nd", "type", "data"])),
                    }
                }

                let nd = nd.ok_or_else(|| de::Error::missing_field("nd"))?;
                let numpy_dtype = numpy_dtype.ok_or_else(|| de::Error::missing_field("type"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                if nd {
                    return Err(de::Error::custom("nd should be false for numpy scalars"));
                }

                // we only support primitive numeric types for now

                match numpy_dtype {
                    DType::String(dtype) => {
                        match dtype.as_str() {
                            // convert through u8 to conform to NumPy's serialization behavior of booleans
                            "|b1" => TryInto::<[u8; 1]>::try_into(data.into_vec())
                                .map(|bytes| Scalar::Bool(bytes[0] != 0))
                                .map_err(|_| de::Error::custom("Invalid data for bool")),
                            "|u1" => TryInto::<[u8; 1]>::try_into(data.into_vec())
                                .map(|bytes| Scalar::U8(bytes[0]))
                                .map_err(|_| de::Error::custom("Invalid data for u8")),
                            "|i1" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I8(i8::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i8")),
                            "<u2" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::U16(u16::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for u16")),
                            "<i2" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I16(i16::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i16")),
                            "<f2" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::F16(f16::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for f16")),
                            "<u4" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::U32(u32::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for u32")),
                            "<i4" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I32(i32::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i32")),
                            "<f4" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::F32(f32::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for f32")),
                            "<u8" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::U64(u64::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for u64")),
                            "<i8" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I64(i64::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i64")),
                            "<f8" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::F64(f64::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for f64")),
                            _ => Ok(Scalar::Unsupported),
                        }
                    }
                    DType::Array(_) => Ok(Scalar::Unsupported),
                }
            }
        }

        deserializer.deserialize_map(ScalarVisitor)
    }
}

// impl Serialize

impl Serialize for Scalar {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(3))?;

        state.serialize_entry(Bytes::new(b"nd"), &false)?;

        match self {
            // convert through u8 to conform to NumPy's serialization behavior of booleans
            Scalar::Bool(val) => serialize_value(&mut state, "|b1", &[*val as u8]),
            Scalar::U8(val) => serialize_value(&mut state, "|u1", &[*val]),
            Scalar::I8(val) => serialize_value(&mut state, "|i1", &val.to_le_bytes()),
            Scalar::U16(val) => serialize_value(&mut state, "<u2", &val.to_le_bytes()),
            Scalar::I16(val) => serialize_value(&mut state, "<i2", &val.to_le_bytes()),
            Scalar::F16(val) => serialize_value(&mut state, "<f2", &val.to_le_bytes()),
            Scalar::U32(val) => serialize_value(&mut state, "<u4", &val.to_le_bytes()),
            Scalar::I32(val) => serialize_value(&mut state, "<i4", &val.to_le_bytes()),
            Scalar::F32(val) => serialize_value(&mut state, "<f4", &val.to_le_bytes()),
            Scalar::U64(val) => serialize_value(&mut state, "<u8", &val.to_le_bytes()),
            Scalar::I64(val) => serialize_value(&mut state, "<i8", &val.to_le_bytes()),
            Scalar::F64(val) => serialize_value(&mut state, "<f8", &val.to_le_bytes()),
            Scalar::Unsupported => {
                return Err(serde::ser::Error::custom("Unsupported numpy dtype"));
            }
        }?;

        state.end()
    }
}

fn serialize_value<S>(state: &mut S, type_str: &str, val: &[u8]) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    state.serialize_entry(Bytes::new(b"type"), type_str)?;
    state.serialize_entry(Bytes::new(b"data"), Bytes::new(val))
}

/***********************************************************************************************/
// NDArray

use ndarray::{Array, IxDyn};
use std::mem;

#[derive(thiserror::Error, Debug)]
enum NDArrayError {
    #[error("InvalidDataLength: {0}")]
    InvalidDataLength(String),

    #[error("ArrayShapeError: {0}")]
    ArrayShapeError(ndarray::ShapeError),
}

// impl Deserialize

impl<'de> Deserialize<'de> for NDArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NDArrayVisitor;

        impl<'de> Visitor<'de> for NDArrayVisitor {
            type Value = NDArray;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a numpy array in msgpack format")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut nd: Option<bool> = None;
                let mut numpy_dtype: Option<DType> = None;
                let mut kind: Option<ByteBuf> = None;
                let mut shape: Option<Vec<usize>> = None;
                let mut data: Option<ByteBuf> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "nd" => nd = Some(map.next_value()?),
                        "type" => numpy_dtype = Some(map.next_value()?),
                        "kind" => kind = Some(map.next_value()?),
                        "shape" => shape = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => {
                            return Err(de::Error::unknown_field(
                                key,
                                &["nd", "type", "kind", "shape", "data"],
                            ))
                        }
                    }
                }

                let nd = nd.ok_or_else(|| de::Error::missing_field("nd"))?;
                let numpy_dtype = numpy_dtype.ok_or_else(|| de::Error::missing_field("type"))?;
                let _kind = kind.ok_or_else(|| de::Error::missing_field("kind"))?;
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                if !nd {
                    return Err(de::Error::custom("nd should be true for numpy arrays"));
                }

                let shape = IxDyn(&shape);

                // we only support primitive numeric types for now

                match numpy_dtype {
                    DType::String(dtype) => {
                        match dtype.as_str() {
                            // convert through u8 to conform to NumPy's serialization behavior of booleans
                            "|b1" => Array::from_shape_vec(
                                shape,
                                data.into_iter().map(|v| v != 0).collect(),
                            )
                            .map(NDArray::Bool)
                            .map_err(de::Error::custom),
                            "|u1" => Array::from_shape_vec(shape, data.into_vec())
                                .map(NDArray::U8)
                                .map_err(de::Error::custom),
                            "|i1" => create_ndarray_from_transmution::<i8>(data.into_vec(), shape)
                                .map(NDArray::I8)
                                .map_err(de::Error::custom),
                            "<u2" => create_ndarray_from_transmution::<u16>(data.into_vec(), shape)
                                .map(NDArray::U16)
                                .map_err(de::Error::custom),
                            "<i2" => create_ndarray_from_transmution::<i16>(data.into_vec(), shape)
                                .map(NDArray::I16)
                                .map_err(de::Error::custom),
                            "<f2" => create_ndarray_from_transmution::<f16>(data.into_vec(), shape)
                                .map(NDArray::F16)
                                .map_err(de::Error::custom),
                            "<u4" => create_ndarray_from_transmution::<u32>(data.into_vec(), shape)
                                .map(NDArray::U32)
                                .map_err(de::Error::custom),
                            "<i4" => create_ndarray_from_transmution::<i32>(data.into_vec(), shape)
                                .map(NDArray::I32)
                                .map_err(de::Error::custom),
                            "<f4" => create_ndarray_from_transmution::<f32>(data.into_vec(), shape)
                                .map(NDArray::F32)
                                .map_err(de::Error::custom),
                            "<u8" => create_ndarray_from_transmution::<u64>(data.into_vec(), shape)
                                .map(NDArray::U64)
                                .map_err(de::Error::custom),
                            "<i8" => create_ndarray_from_transmution::<i64>(data.into_vec(), shape)
                                .map(NDArray::I64)
                                .map_err(de::Error::custom),
                            "<f8" => create_ndarray_from_transmution::<f64>(data.into_vec(), shape)
                                .map(NDArray::F64)
                                .map_err(de::Error::custom),
                            _ => Ok(NDArray::Unsupported),
                        }
                    }
                    DType::Array(_) => Ok(NDArray::Unsupported),
                }
            }
        }

        deserializer.deserialize_map(NDArrayVisitor)
    }
}

/// Creates an n-dimensional array from raw byte data by transmuting it to the specified type.
///
/// # Type Parameters
///
/// * `T`: The target numeric type for transmutation (e.g., f32, i64).
///
/// # Arguments
///
/// * `data`: Raw bytes to be transmuted and reshaped.
/// * `shape`: The desired shape of the output array.
///
/// # Returns
///
/// An n-dimensional array of type `T` with the specified shape, or an error.
///
/// # Errors
///
/// Returns an error if:
/// * Transmutation fails (e.g., data length isn't a multiple of `size_of::<T>()`).
/// * Specified shape doesn't match the transmuted data length.
///
/// # Safety
///
/// Caller must ensure:
/// * Input data represents valid values of type `T`.
/// * Data length is a multiple of `size_of::<T>()`.
/// * Memory layout of `T` is compatible with the original data.
fn create_ndarray_from_transmution<T>(
    data: Vec<u8>,
    shape: ndarray::IxDyn,
) -> Result<Array<T, IxDyn>, NDArrayError> {
    let transmuted = unsafe { transmute_vec(data) }.ok_or_else(|| {
        NDArrayError::InvalidDataLength(format!(
            "Invalid data length for {} transmutation",
            std::any::type_name::<T>()
        ))
    })?;

    Array::from_shape_vec(shape, transmuted).map_err(|e| NDArrayError::ArrayShapeError(e))
}

/// Transmutes a `Vec<u8>` into a `Vec<T>`.
///
/// We could use vec_into_raw_parts when that stabilizes.
/// e.g. let (ptr, len, cap) = data.into_raw_parts();
///
/// # Safety
///
/// This function is unsafe because it assumes that:
/// - The input `data` is correctly formatted and aligned for type `T`.
///
/// # Type Parameters
///
/// * `T`: The target numeric type for transmutation (e.g., f32, i64).
///
/// # Arguments
///
/// * `data` - A `Vec<u8>` containing the raw byte data.
///
/// # Returns
///
/// Returns `Some(Vec<T>)` containing the transmuted data if successful,
/// or `None` if the input length is not a multiple of `size_of::<T>()`.
unsafe fn transmute_vec<T>(mut data: Vec<u8>) -> Option<Vec<T>> {
    if data.len() % mem::size_of::<T>() != 0 {
        return None;
    }

    let ptr = data.as_mut_ptr() as *mut T;
    let len = data.len() / mem::size_of::<T>();
    let capacity = data.capacity() / mem::size_of::<T>();

    // Ensure we don't drop the original vector's memory
    mem::forget(data);

    Some(Vec::from_raw_parts(ptr, len, capacity))
}

// impl Serialize

impl Serialize for NDArray {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(5))?;

        state.serialize_entry(Bytes::new(b"nd"), &true)?;

        match self {
            // convert through u8 to conform to NumPy's serialization behavior of booleans
            NDArray::Bool(arr) => serialize_array(&mut state, "|b1", &arr.mapv(|v| v as u8)),
            NDArray::U8(arr) => serialize_array(&mut state, "|u1", arr),
            NDArray::I8(arr) => serialize_array(&mut state, "|i1", arr),
            NDArray::U16(arr) => serialize_array(&mut state, "<u2", arr),
            NDArray::I16(arr) => serialize_array(&mut state, "<i2", arr),
            NDArray::F16(arr) => serialize_array(&mut state, "<f2", arr),
            NDArray::U32(arr) => serialize_array(&mut state, "<u4", arr),
            NDArray::I32(arr) => serialize_array(&mut state, "<i4", arr),
            NDArray::F32(arr) => serialize_array(&mut state, "<f4", arr),
            NDArray::U64(arr) => serialize_array(&mut state, "<u8", arr),
            NDArray::I64(arr) => serialize_array(&mut state, "<i8", arr),
            NDArray::F64(arr) => serialize_array(&mut state, "<f8", arr),
            NDArray::Unsupported => {
                return Err(serde::ser::Error::custom("Unsupported numpy dtype"));
            }
        }?;

        state.end()
    }
}

fn serialize_array<S, T>(
    state: &mut S,
    type_str: &str,
    arr: &Array<T, ndarray::IxDyn>,
) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    state.serialize_entry(Bytes::new(b"type"), type_str)?;
    state.serialize_entry(Bytes::new(b"kind"), Bytes::new(b""))?;
    state.serialize_entry(Bytes::new(b"shape"), &arr.shape())?;

    let data = unsafe { transmute_to_bytes(arr) };
    state.serialize_entry(Bytes::new(b"data"), Bytes::new(data))
}

/// Converts an n-dimensional array to a byte slice without copying.
///
/// # Safety
///
/// This function is unsafe because:
/// - It assumes the memory layout of `T` is contiguous with no padding.
///
/// # Type Parameters
///
/// * `T`: The target numeric type for transmutation (e.g., f32, i64).
///
/// # Arguments
///
/// * `arr` - A reference to an n-dimensional array of type `T`.
///
/// # Returns
///
/// A byte slice (`&[u8]`) representing the raw memory of the input array.
unsafe fn transmute_to_bytes<T>(arr: &Array<T, ndarray::IxDyn>) -> &[u8] {
    let ptr = arr.as_ptr() as *const u8;
    let len = arr.len() * mem::size_of::<T>();
    std::slice::from_raw_parts(ptr, len)
}

/*********************************************************************************/
// tests

#[cfg(test)]
mod tests {
    // use super::*;
    use crate::core::{NDArray, Scalar};
    use half::f16;
    use ndarray::Array;
    use serde::{Deserialize, Serialize};

    fn serialize_and_deserialize<T: Serialize + for<'de> Deserialize<'de>>(value: &T) -> T {
        let buf = rmp_serde::to_vec_named(value).unwrap();
        rmp_serde::from_slice(&buf).unwrap()
    }

    #[test]
    fn test_scalar_serialization() {
        let cases = vec![
            Scalar::Bool(true),
            Scalar::U8(255),
            Scalar::I8(-128),
            Scalar::U16(65535),
            Scalar::I16(-32768),
            Scalar::F16(f16::from_f32(1.0)),
            Scalar::U32(4294967295),
            Scalar::I32(-2147483648),
            Scalar::F32(1.0),
            Scalar::U64(18446744073709551615),
            Scalar::I64(-9223372036854775808),
            Scalar::F64(1.0),
        ];

        for scalar in cases {
            let deserialized: Scalar = serialize_and_deserialize(&scalar);
            assert_eq!(deserialized, scalar);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn test_ndarray_serialization() {
        let cases = vec![
            NDArray::Bool(Array::from_vec(vec![true, false]).into_dyn()),
            NDArray::U8(Array::from_vec(vec![1, 2, 3]).into_dyn()),
            NDArray::I8(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::U16(Array::from_vec(vec![1, 2, 3]).into_dyn()),
            NDArray::I16(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::F16(Array::from_vec(vec![1.0, 2.0]).into_dyn().mapv(f16::from_f32)),
            NDArray::U32(Array::from_vec(vec![1, 2, 3]).into_dyn()),
            NDArray::I32(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::F32(Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn()),
            NDArray::U64(Array::from_vec(vec![1, 2]).into_dyn()),
            NDArray::I64(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::F64(Array::from_vec(vec![1.0, 2.0]).into_dyn()),
        ];

        for ndarray in cases {
            let deserialized: NDArray = serialize_and_deserialize(&ndarray);
            assert_eq!(deserialized, ndarray);
        }
    }
}
