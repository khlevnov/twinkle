use std::ops::{Index, Range};
use std::rc::Rc;
use std::slice::SliceIndex;

#[derive(Debug)]
pub(crate) struct Storage<T> {
    pub(super) data: Rc<[T]>,
    pub(super) range: Range<usize>,
}

impl<T> Storage<T> {
    fn to<S: Copy>(&self) -> Storage<S> where T: Copy + Into<S> {
        self.data.iter()
            .map(|x| (*x).into())
            .collect::<Vec<S>>()
            .into_boxed_slice()
            .into()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Clone for Storage<T> {
    fn clone(&self) -> Self {
        Storage {
            data: Rc::clone(&self.data),
            range: self.range.clone(),
        }
    }
}

impl<T: Copy, D: AsRef<[T]>> From<D> for Storage<T> {
    fn from(data: D) -> Self {
        Storage {
            range: 0..data.as_ref().len(),
            data: data.as_ref().into(),
        }
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for Storage<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.data[self.range.clone()].index(index)
    }
}
