use crate::tensor::tensor::Tensor;

trait BinaryOp<T> {
    fn binary_op(&self, rhs: &Tensor<T>) -> Tensor<T>;
}

impl<T: Copy> BinaryOp<T> for Tensor<T> {
    fn binary_op(&self, rhs: &Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape(), rhs.shape());

        todo!()
    }
}
