use crate::tensor::shape::Shape;
use crate::tensor::storage::Storage;

#[derive(Debug)]
pub struct Tensor<T> {
    pub(super) storage: Storage<T>,
    pub(super) shape: Shape,
}

impl<T: Copy> Tensor<T> {
    pub fn len(&self) -> usize {
        assert_ne!(self.shape.shape().len(), 0, "len() of a 0-d tensor");
        self.shape.shape()[0]
    }

    pub fn numel(&self) -> usize {
        self.storage.len()
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.shape.strides()
    }

    // TODO: Support single -1 dim
    pub fn reshape<S: AsRef<[usize]>>(&self, shape: S) -> Tensor<T> {
        if self.is_contiguous() {
            Tensor {
                storage: self.storage.clone(),
                shape: self.shape.reshape(shape),
            }
        } else {
            let contiguous = self.contiguous();
            Tensor {
                storage: contiguous.storage.clone(),
                shape: contiguous.shape.reshape(shape),
            }
        }
    }

    fn is_subspace<S: AsRef<[usize]>>(&self, _shape: S) -> bool {
        false
    }

    pub fn transpose<A: AsRef<[usize]>>(&self, axis: A) -> Tensor<T> {
        assert_eq!(self.shape.shape().len(), axis.as_ref().len(), "axis don't match tensor");
        Tensor {
            storage: self.storage.clone(),
            shape: self.shape.transpose(axis),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous()
    }

    pub fn contiguous(&self) -> Tensor<T> {
        self.shape
            .indices()
            .iter()
            .map(|i| self.index_to_storage_index(i))
            .map(|i| self.storage[i])
            .collect::<Vec<T>>()
            .into()
    }

    pub fn item(&self) -> T {
        assert_eq!(self.numel(), 1, "only one element tensors can be converted to scalars");
        todo!()
    }

    pub(super) fn index_to_storage_index<I: AsRef<[usize]>>(&self, index: I) -> usize {
        index.as_ref()
            .iter()
            .zip(self.strides())
            .map(|(i, stride)| i * stride)
            .sum()
    }

    pub fn expand<S: AsRef<[usize]>>(&self, shape: S) -> Tensor<T> {
        Tensor {
            storage: self.storage.clone(),
            shape: self.shape.expand(shape),
        }
    }

    // fn is_broadcastable<S: AsRef<[usize]>>(&self, shape: S) -> Result<(), E> {
    //     let shape = shape.as_ref();
    //     if self.shape().len() > shape.len() {
    //         // return false;
    //     }
    //
    //     let iter = self.shape()
    //         .iter()
    //         .cloned()
    //         .rev()
    //         .chain(vec![1usize; shape.len() - self.shape().len()])
    //         .zip(shape.iter().cloned().rev())
    //         .all(|(a, b)| a == b || a == 1 || b == 1);
    //
    //     todo!()
    //     // for (a, b) in iter {
    //     //     //
    //     //     "The expanded size of the tensor ({}) must match the existing size ({}) at non-singleton dimension 3. Target sizes: [2, 3, 2, 5].  Tensor sizes: [3, 1, 4]"
    //     // }
    // }
}

impl<T: Copy> From<T> for Tensor<T> {
    fn from(data: T) -> Self {
        Tensor {
            storage: vec![data].into(),
            shape: Shape::default(),
        }
    }
}

impl<T: Copy> From<&[T]> for Tensor<T> {
    fn from(data: &[T]) -> Self {
        Tensor {
            shape: vec![data.as_ref().len()].into(),
            storage: data.into(),
        }
    }
}

impl<T: Copy> From<Vec<T>> for Tensor<T> {
    fn from(data: Vec<T>) -> Self {
        Tensor {
            shape: vec![data.len()].into(),
            storage: data.into(),
        }
    }
}


