use crate::tensor::index::Index;
use crate::tensor::tensor::Tensor;

pub struct Iter<T> {
    tensor: Tensor<T>,
    next: usize,
}

impl<T: Copy> Iterator for Iter<T>
    where Tensor<T>: Index<T>
{
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.tensor.len() {
            return None
        } else {
            let index = self.next;
            self.next += 1;
            Some(self.tensor.index(&[index]))
        }
    }
}

impl<T: Copy> From<Tensor<T>> for Iter<T> {
    fn from(tensor: Tensor<T>) -> Self {
        assert_ne!(tensor.len(), 0, "iteration over a 0-d tensor");
        Iter {
            tensor,
            next: 0,
        }
    }
}

impl<T: Copy> IntoIterator for Tensor<T> {
    type Item = Tensor<T>;
    type IntoIter = Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }
}
