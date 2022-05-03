// use std::iter::zip;
// use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Debug)]
pub(crate) enum TensorData<'a, T>{
    Scalar(T),
    Owned(Vec<T>),
    Borrowed(&'a [T]),
}

use self::TensorData::*;

pub(crate) struct TensorIter {
    iter: i32,
}

// impl<'a, T> Iterator for TensorIter<'a, T> {
//     type Item = &'a T;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next()
//     }
// }

impl<'a, T> TensorData<'a, T> {
    fn numel(&self) -> usize {
        match self {
            Scalar(_) => 1,
            Owned(data) => data.len(),
            Borrowed(data) => data.len(),
        }
    }

    fn iter(&self) -> TensorIter {
        TensorIter { iter: 0 }
    }

//     fn as_slice(&self) -> &[T] {
//         match self {
//             Scalar(_) => panic!("invalid index of a 0-dim tensor. Use `Tensor::item()`"),
//             Owned(x) => x.as_slice(),
//             Borrowed(x) => x,
//         }
//     }
}

#[derive(Clone, Debug)]
pub struct Tensor<'a, T>
    where T: Clone,
{
    pub(crate) data: TensorData<'a, T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) axis: Vec<usize>,
}

impl<'a, T> Tensor<'a, T>
    where T: Clone
{
//     pub fn binary_op<Out, F>(&self, other: &Tensor<'a, T>, op: F) -> Tensor<'a, Out>
//         where Out: TensorItem,
//               F: Fn(T, T) -> Out
//     {
//         let (mut tensor, shape): (Tensor<'a, Out>, &[usize]) = match (&self.data, &other.data) {
//             (Scalar(lhs), Scalar(rhs)) => return op(*lhs, *rhs).into(),
//             (Scalar(lhs), rhs @ _) => {
//                 let data = rhs
//                     .iter()
//                     .map(|x| op(*lhs, *x))
//                     .collect::<Vec<Out>>();
//                 (data.into(), other.shape())
//             },
//             (lhs @ _, Scalar(rhs)) => {
//                 let data = lhs
//                     .iter()
//                     .map(|x| op(*x, *rhs))
//                     .collect::<Vec<Out>>();
//                 (data.into(), self.shape())
//             },
//             (lhs @ _, rhs @ _) => {
//                 let data = zip(lhs.iter(), rhs.iter())
//                     .map(|(a, b)| op(*a, *b))
//                     .collect::<Vec<Out>>();
//                 (data.into(), self.shape())
//             },
//         };
//         tensor.shape = shape.to_vec();
//         tensor
//     }
//
//     pub fn unary_op<Out, F>(&self, op: F) -> Tensor<'a, Out>
//         where Out: TensorItem,
//               F: Fn(T) -> Out
//     {
//         let mut tensor: Tensor<'a, Out> = match &self.data {
//             Scalar(x) => return op(*x).into(),
//             x => x.iter()
//                 .map(|x| op(*x))
//                 .collect::<Vec<Out>>()
//                 .into(),
//         };
//         tensor.shape = self.shape.clone();
//         tensor
//     }
//
//     fn elementwise_eq(&self, other: &Self) -> Tensor<'a, bool> {
//         self.binary_op(other, |a, b| a == b)
//     }
//
//     fn elementwise_ne(&self, other: &Self) -> Tensor<'a, bool> {
//         self.binary_op(other, |a, b| a != b)
//     }
//
//     fn elementwise_gt(&self, other: &Self) -> Tensor<'a, bool> {
//         self.binary_op(other, |a, b| a > b)
//     }
//
//     fn elementwise_lt(&self, other: &Self) -> Tensor<'a, bool> {
//         self.binary_op(other, |a, b| a < b)
//     }
}