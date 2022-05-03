// use std::iter::zip;
// use std::ops::{Add, Div, Mul, Neg, Sub};
// use crate::tensor::*;
// use crate::TensorData::*;
//
// impl<'a, T: TensorItem + Add<Output = T>> Add<&T> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn add(self, rhs: &T) -> Self::Output {
//         self.binary_op(&Tensor::from(*rhs), |a, b| a + b)
//     }
// }
//
// impl<'a, T: TensorItem + Add<Output = T>> Add<&Tensor<'a, T>> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn add(self, rhs: &Tensor<'a, T>) -> Self::Output {
//         self.binary_op(&rhs, |a, b| a + b)
//     }
// }
//
// impl<'a, T: TensorItem + Mul<Output = T>> Mul<&T> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn mul(self, rhs: &T) -> Self::Output {
//         self.binary_op(&Tensor::from(*rhs), |a, b| a * b)
//     }
// }
//
// impl<'a, T: TensorItem + Mul<Output = T>> Mul<&Tensor<'a, T>> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn mul(self, rhs: &Tensor<'a, T>) -> Self::Output {
//         self.binary_op(&rhs, |a, b| a * b)
//     }
// }
//
// impl<'a, T: TensorItem + Sub<Output = T>> Sub<&T> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn sub(self, rhs: &T) -> Self::Output {
//         self.binary_op(&Tensor::from(*rhs), |a, b| a - b)
//     }
// }
//
// impl<'a, T: TensorItem + Sub<Output = T>> Sub<&Tensor<'a, T>> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn sub(self, rhs: &Tensor<'a, T>) -> Self::Output {
//         self.binary_op(&rhs, |a, b| a - b)
//     }
// }
//
// impl<'a, T: TensorItem + Div<Output = T>> Div<&T> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn div(self, rhs: &T) -> Self::Output {
//         self.binary_op(&Tensor::from(*rhs), |a, b| a / b)
//     }
// }
//
// impl<'a, T: TensorItem + Div<Output = T>> Div<&Tensor<'a, T>> for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn div(self, rhs: &Tensor<'a, T>) -> Self::Output {
//         self.binary_op(&rhs, |a, b| a / b)
//     }
// }
//
// impl<'a, T: TensorItem + Sub<Output = T> + Default> Neg for &Tensor<'a, T> {
//     type Output = Tensor<'a, T>;
//
//     fn neg(self) -> Self::Output {
//         self.unary_op(|x| T::default() - x)
//     }
// }
//
// pub trait Dot {
//     type Output;
//
//     fn dot(&self, other: &Self) -> Self::Output;
// }
//
// impl<'a, T> Dot for Tensor<'a, T>
//     where T: TensorItem + Mul<Output = T>
// {
//     type Output = Tensor<'a, T>;
//
//     fn dot(&self, other: &Self) -> Self::Output {
//         assert_eq!(self.shape, other.shape, "The shape of tensor a ({:?}) \
//             must match the shape of tensor b ({:?})", self.shape, other.shape);
//         assert_eq!(self.shape.len(), 1, "1D tensors expected");
//
//         zip(self.data.iter(), other.data.iter())
//             .map(|(a, b)| *a * *b)
//             .reduce(|acc, x| acc * x)
//             .unwrap()
//             .into()
//     }
// }
//
// pub trait MatMul {
//     type Output;
//
//     fn mm(&self, other: &Self) -> Self::Output;
// }
//
// impl<'a, T> MatMul for Tensor<'a, T>
//     where T: TensorItem + Mul<Output = T>
// {
//     type Output = Tensor<'a, T>;
//
//     fn mm(&self, other: &Self) -> Self::Output {
//         // let result = Tensor::<f64>::from(vec![0f64; self.shape()[0] * other.shape()[1]]);
//         zip(self.data.iter(), other.data.iter())
//             .map(|(a, b)| *a * *b)
//             .reduce(|acc, x| acc * x)
//             .unwrap()
//             .into()
//     }
// }
//
// pub trait Reduce {
//     type Output;
//
//     fn mean(&self) -> Self::Output;
//
//     fn sum(&self) -> Self::Output;
// }
//
// impl<'a, T> Reduce for Tensor<'a, T>
//     where T: TensorItem + Default + Add<Output = T> + Div<Output = T> + From<u32>
// {
//     type Output = Tensor<'a, T>;
//
//     fn mean(&self) -> Self::Output {
//         match self.data {
//             Scalar(_) => self.clone(),
//             _ => {
//                 let mean = self.data.iter()
//                     .fold(T::default(), |acc, x| acc + *x);
//                 (mean / T::try_from(u32::try_from(self.len()).unwrap()).unwrap()).into()
//             }
//         }
//     }
//
//     fn sum(&self) -> Self::Output {
//         match self.data {
//             Scalar(_) => self.clone(),
//             _ => self.data
//                 .iter()
//                 .fold(T::default(), |acc, x| acc + *x)
//                 .into()
//         }
//     }
// }
//
// pub trait Pow {
//     type Output;
//
//     fn pow(self, power: i32) -> Self::Output;
// }
//
// impl<'a, T> Pow for &Tensor<'a, T>
//     where T: TensorItem + Mul<Output = T>
// {
//     type Output = Tensor<'a, T>;
//
//     fn pow(self, power: i32) -> Self::Output {
//         let pow = |x| {
//             let mut result = x;
//             for _ in 0..(power - 1) {
//                 result = result * x;
//             }
//             result
//         };
//         self.unary_op(pow)
//     }
// }
//
// pub trait Argmax {
//     fn argmax(&self) -> Tensor<usize>;
// }
//
// impl<'a, T: TensorItem> Argmax for Tensor<'a, T> {
//     fn argmax(&self) -> Tensor<usize> {
//         match &self.data {
//             Scalar(_) => 0.into(),
//             _ => {
//                 let mut max = (0usize, self.data.iter().next().unwrap());
//                 for (idx, item) in self.data.iter().enumerate() {
//                     if item > max.1 {
//                         max = (idx, item)
//                     }
//                 }
//                 max.0.into()
//             }
//         }
//     }
// }
//
// pub trait Transpose {
//     fn T(&self) -> Self;
// }
//
// impl<'a, T: TensorItem> Transpose for Tensor<'a, T> {
//     fn T(&self) -> Self {
//         // Add axis
//         assert_eq!(self.shape.len(), 2, "self must be a matrix");
//         self.clone()
//     }
// }
