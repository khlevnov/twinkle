// use std::ops::{Add, Div, Mul, Sub};
// use crate::tensor::*;
// use crate::math::Pow;
// use crate::math::Reduce;
//
// fn reduce_loss<'a, T>(loss: Tensor<'a, T>, reduction: &str) -> Tensor<'a, T>
//     where T: TensorItem + Add<Output = T> + Div<Output = T> + From<u32>
// {
//     match reduction {
//         "none" => loss,
//         "mean" => loss.mean(),
//         "sum" => loss.sum(),
//         _ => panic!("{} is not a valid value for reduction", reduction),
//     }
// }
//
// pub fn mse_loss<'a, T>(
//     y_pred: &Tensor<'a, T>,
//     y_true: &Tensor<'a, T>,
//     reduction: &str,
// ) -> Tensor<'a, T>
//     where T: TensorItem,
//           T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<u32>
// {
//     let loss = (y_pred - y_true).pow(2);
//     reduce_loss(loss, reduction)
// }
//
// pub fn binary_cross_entropy<'a, T>(
//     y_pred: &Tensor<'a, T>,
//     y_true: &Tensor<'a, T>,
//     weight: &Tensor<'a, T>,
//     reduction: &str,
// ) -> Tensor<'a, T>
//     where T: TensorItem,
//           T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<u32>
// {
//     let loss = unimplemented!();
//     reduce_loss(loss, reduction)
// }
