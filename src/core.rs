use half::f16;
use num_traits::{NumCast, ToPrimitive};

/*********************************************************************************/
// Scalar

/// De-/serialization target for a NumPy scalar
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    F16(f16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Unsupported,
}

impl Scalar {
    pub fn to_bool(&self) -> Option<bool> {
        match self {
            Scalar::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn to_u8(&self) -> Option<u8> {
        self.to()
    }

    pub fn to_i8(&self) -> Option<i8> {
        self.to()
    }

    pub fn to_u16(&self) -> Option<u16> {
        self.to()
    }

    pub fn to_i16(&self) -> Option<i16> {
        self.to()
    }

    pub fn to_f16(&self) -> Option<f16> {
        self.to()
    }

    pub fn to_u32(&self) -> Option<u32> {
        self.to()
    }

    pub fn to_i32(&self) -> Option<i32> {
        self.to()
    }

    pub fn to_f32(&self) -> Option<f32> {
        self.to()
    }

    pub fn to_u64(&self) -> Option<u64> {
        self.to()
    }

    pub fn to_i64(&self) -> Option<i64> {
        self.to()
    }

    pub fn to_f64(&self) -> Option<f64> {
        self.to()
    }

    fn to<T: NumCast>(&self) -> Option<T> {
        match self {
            // bool doesn't implement ToPrimitive, so we need to convert it to u8 first
            Scalar::Bool(v) => NumCast::from(*v as u8),
            Scalar::U8(v) => NumCast::from(*v),
            Scalar::I8(v) => NumCast::from(*v),
            Scalar::U16(v) => NumCast::from(*v),
            Scalar::I16(v) => NumCast::from(*v),
            Scalar::F16(v) => NumCast::from(*v),
            Scalar::U32(v) => NumCast::from(*v),
            Scalar::I32(v) => NumCast::from(*v),
            Scalar::F32(v) => NumCast::from(*v),
            Scalar::U64(v) => NumCast::from(*v),
            Scalar::I64(v) => NumCast::from(*v),
            Scalar::F64(v) => NumCast::from(*v),
            Scalar::Unsupported => None,
        }
    }
}

/*********************************************************************************/
// NDArray

use ndarray::{Array, IxDyn};

/// De-/serialization target for a NumPy array that uses owned Array for deserialization
#[derive(Debug, Clone, PartialEq)]
pub enum NDArray {
    Bool(Array<bool, IxDyn>),
    U8(Array<u8, IxDyn>),
    I8(Array<i8, IxDyn>),
    U16(Array<u16, IxDyn>),
    I16(Array<i16, IxDyn>),
    F16(Array<f16, IxDyn>),
    U32(Array<u32, IxDyn>),
    I32(Array<i32, IxDyn>),
    F32(Array<f32, IxDyn>),
    U64(Array<u64, IxDyn>),
    I64(Array<i64, IxDyn>),
    F64(Array<f64, IxDyn>),
    Unsupported,
}

impl NDArray {
    pub fn into_bool_array(self) -> Option<Array<bool, IxDyn>> {
        match self {
            NDArray::Bool(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn into_u8_array(self) -> Option<Array<u8, IxDyn>> {
        match self {
            NDArray::U8(arr) => Some(arr),
            _ => self.convert_into::<u8>(),
        }
    }

    pub fn into_i8_array(self) -> Option<Array<i8, IxDyn>> {
        match self {
            NDArray::I8(arr) => Some(arr),
            _ => self.convert_into::<i8>(),
        }
    }

    pub fn into_u16_array(self) -> Option<Array<u16, IxDyn>> {
        match self {
            NDArray::U16(arr) => Some(arr),
            _ => self.convert_into::<u16>(),
        }
    }

    pub fn into_i16_array(self) -> Option<Array<i16, IxDyn>> {
        match self {
            NDArray::I16(arr) => Some(arr),
            _ => self.convert_into::<i16>(),
        }
    }

    pub fn into_f16_array(self) -> Option<Array<f16, IxDyn>> {
        match self {
            NDArray::F16(arr) => Some(arr),
            _ => self.convert_into::<f16>(),
        }
    }

    pub fn into_u32_array(self) -> Option<Array<u32, IxDyn>> {
        match self {
            NDArray::U32(arr) => Some(arr),
            _ => self.convert_into::<u32>(),
        }
    }

    pub fn into_i32_array(self) -> Option<Array<i32, IxDyn>> {
        match self {
            NDArray::I32(arr) => Some(arr),
            _ => self.convert_into::<i32>(),
        }
    }

    pub fn into_f32_array(self) -> Option<Array<f32, IxDyn>> {
        match self {
            NDArray::F32(arr) => Some(arr),
            _ => self.convert_into::<f32>(),
        }
    }

    pub fn into_u64_array(self) -> Option<Array<u64, IxDyn>> {
        match self {
            NDArray::U64(arr) => Some(arr),
            _ => self.convert_into::<u64>(),
        }
    }

    pub fn into_i64_array(self) -> Option<Array<i64, IxDyn>> {
        match self {
            NDArray::I64(arr) => Some(arr),
            _ => self.convert_into::<i64>(),
        }
    }

    pub fn into_f64_array(self) -> Option<Array<f64, IxDyn>> {
        match self {
            NDArray::F64(arr) => Some(arr),
            _ => self.convert_into::<f64>(),
        }
    }

    fn convert_into<T: NumCast + Copy>(self) -> Option<Array<T, IxDyn>> {
        match self {
            NDArray::Bool(arr) => Self::convert_bool_array(arr),
            NDArray::U8(arr) => Self::convert_array(arr),
            NDArray::I8(arr) => Self::convert_array(arr),
            NDArray::U16(arr) => Self::convert_array(arr),
            NDArray::I16(arr) => Self::convert_array(arr),
            NDArray::F16(arr) => Self::convert_array(arr),
            NDArray::U32(arr) => Self::convert_array(arr),
            NDArray::I32(arr) => Self::convert_array(arr),
            NDArray::F32(arr) => Self::convert_array(arr),
            NDArray::U64(arr) => Self::convert_array(arr),
            NDArray::I64(arr) => Self::convert_array(arr),
            NDArray::F64(arr) => Self::convert_array(arr),
            NDArray::Unsupported => None,
        }
    }

    fn convert_array<S: Copy + ToPrimitive, T: NumCast>(
        arr: Array<S, IxDyn>,
    ) -> Option<Array<T, IxDyn>> {
        let raw_dim = arr.raw_dim();
        arr.into_iter()
            .map(|v| NumCast::from(v).ok_or(()))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .map(|vec| Array::from_shape_vec(raw_dim, vec).unwrap())
    }

    fn convert_bool_array<T: NumCast>(arr: Array<bool, IxDyn>) -> Option<Array<T, IxDyn>> {
        let raw_dim = arr.raw_dim();
        arr.into_iter()
            .map(|v| NumCast::from(v as u8).ok_or(()))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .map(|vec| Array::from_shape_vec(raw_dim, vec).unwrap())
    }
}

/*********************************************************************************/
// CowNDArray

use ndarray::CowArray;

/// De-/serialization target for a NumPy array that uses CowArray for zero-copy deserialization (when array buffer alignment is good)
#[derive(Debug, Clone, PartialEq)]
pub enum CowNDArray<'a> {
    Bool(CowArray<'a, bool, IxDyn>),
    U8(CowArray<'a, u8, IxDyn>),
    I8(CowArray<'a, i8, IxDyn>),
    U16(CowArray<'a, u16, IxDyn>),
    I16(CowArray<'a, i16, IxDyn>),
    F16(CowArray<'a, f16, IxDyn>),
    U32(CowArray<'a, u32, IxDyn>),
    I32(CowArray<'a, i32, IxDyn>),
    F32(CowArray<'a, f32, IxDyn>),
    U64(CowArray<'a, u64, IxDyn>),
    I64(CowArray<'a, i64, IxDyn>),
    F64(CowArray<'a, f64, IxDyn>),
    Unsupported,
}

impl<'a> CowNDArray<'a> {
    pub fn into_bool_array(self) -> Option<CowArray<'a, bool, IxDyn>> {
        match self {
            CowNDArray::Bool(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn into_u8_array(self) -> Option<CowArray<'a, u8, IxDyn>> {
        match self {
            CowNDArray::U8(arr) => Some(arr),
            _ => self.convert_into::<u8>(),
        }
    }

    pub fn into_i8_array(self) -> Option<CowArray<'a, i8, IxDyn>> {
        match self {
            CowNDArray::I8(arr) => Some(arr),
            _ => self.convert_into::<i8>(),
        }
    }

    pub fn into_u16_array(self) -> Option<CowArray<'a, u16, IxDyn>> {
        match self {
            CowNDArray::U16(arr) => Some(arr),
            _ => self.convert_into::<u16>(),
        }
    }

    pub fn into_i16_array(self) -> Option<CowArray<'a, i16, IxDyn>> {
        match self {
            CowNDArray::I16(arr) => Some(arr),
            _ => self.convert_into::<i16>(),
        }
    }

    pub fn into_f16_array(self) -> Option<CowArray<'a, f16, IxDyn>> {
        match self {
            CowNDArray::F16(arr) => Some(arr),
            // round trip through f32 if not already f16
            _ => self.convert_into::<f16>(),
        }
    }

    pub fn into_u32_array(self) -> Option<CowArray<'a, u32, IxDyn>> {
        match self {
            CowNDArray::U32(arr) => Some(arr),
            _ => self.convert_into::<u32>(),
        }
    }

    pub fn into_i32_array(self) -> Option<CowArray<'a, i32, IxDyn>> {
        match self {
            CowNDArray::I32(arr) => Some(arr),
            _ => self.convert_into::<i32>(),
        }
    }

    pub fn into_f32_array(self) -> Option<CowArray<'a, f32, IxDyn>> {
        match self {
            CowNDArray::F32(arr) => Some(arr),
            _ => self.convert_into::<f32>(),
        }
    }

    pub fn into_u64_array(self) -> Option<CowArray<'a, u64, IxDyn>> {
        match self {
            CowNDArray::U64(arr) => Some(arr),
            _ => self.convert_into::<u64>(),
        }
    }

    pub fn into_i64_array(self) -> Option<CowArray<'a, i64, IxDyn>> {
        match self {
            CowNDArray::I64(arr) => Some(arr),
            _ => self.convert_into::<i64>(),
        }
    }

    pub fn into_f64_array(self) -> Option<CowArray<'a, f64, IxDyn>> {
        match self {
            CowNDArray::F64(arr) => Some(arr),
            _ => self.convert_into::<f64>(),
        }
    }

    fn convert_into<T: NumCast + Copy>(self) -> Option<CowArray<'a, T, IxDyn>> {
        match self {
            CowNDArray::Bool(arr) => Self::convert_bool_array(arr),
            CowNDArray::U8(arr) => Self::convert_array(arr),
            CowNDArray::I8(arr) => Self::convert_array(arr),
            CowNDArray::U16(arr) => Self::convert_array(arr),
            CowNDArray::I16(arr) => Self::convert_array(arr),
            CowNDArray::F16(arr) => Self::convert_array(arr),
            CowNDArray::U32(arr) => Self::convert_array(arr),
            CowNDArray::I32(arr) => Self::convert_array(arr),
            CowNDArray::F32(arr) => Self::convert_array(arr),
            CowNDArray::U64(arr) => Self::convert_array(arr),
            CowNDArray::I64(arr) => Self::convert_array(arr),
            CowNDArray::F64(arr) => Self::convert_array(arr),
            CowNDArray::Unsupported => None,
        }
    }

    fn convert_array<S: Copy + ToPrimitive, T: NumCast>(
        arr: CowArray<S, IxDyn>,
    ) -> Option<CowArray<T, IxDyn>> {
        let raw_dim = arr.raw_dim();
        arr.into_iter()
            .map(|v| NumCast::from(v).ok_or(()))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .map(|vec| Array::from_shape_vec(raw_dim, vec).unwrap().into())
    }

    fn convert_bool_array<T: NumCast>(arr: CowArray<bool, IxDyn>) -> Option<CowArray<T, IxDyn>> {
        let raw_dim = arr.raw_dim();
        arr.into_iter()
            .map(|v| NumCast::from(v as u8).ok_or(()))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .map(|vec| Array::from_shape_vec(raw_dim, vec).unwrap().into())
    }
}
