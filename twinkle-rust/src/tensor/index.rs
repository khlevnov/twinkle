use crate::tensor::storage::Storage;
use crate::tensor::tensor::Tensor;

pub trait Index<T> {
    fn index<I: AsRef<[usize]>>(&self, index: I) -> Tensor<T>;
}

impl<T: Copy> Index<T> for Tensor<T> {
    fn index<I: AsRef<[usize]>>(&self, index: I) -> Tensor<T> {
        let index = index.as_ref();
        assert!(self.shape().len() >= index.len());

        if self.is_contiguous() {
            let from = self.index_to_storage_index(&index);
            let to = from + self.shape()
                .iter()
                .skip(index.len())
                .product::<usize>();

            let mut storage = self.storage.clone();
            storage.range = from..to;

            let shape = self.shape()
                .iter()
                .skip(index.len())
                .cloned()
                .collect::<Vec<usize>>()
                .into();

            return Tensor {
                storage,
                shape,
            }
        }

        let tensor: Tensor<T> = index.as_ref()
            .iter()
            .enumerate()
            .fold(self.shape.indices(), |acc, (i, x)| {
                acc.into_iter()
                    .filter(|index| index[i] == *x)
                    .collect::<Vec<Vec<usize>>>()
            })
            .into_iter()
            .map(|i| self.index_to_storage_index(i))
            .map(|i| self.storage[i])
            .collect::<Vec<T>>()
            .into();

        let shape = self.shape()
            .iter()
            .skip(index.as_ref().len())
            .cloned()
            .collect::<Vec<usize>>();

        tensor.reshape(shape)
    }
}
