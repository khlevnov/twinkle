#[derive(Debug)]
pub(super) struct Shape {
    axis: Box<[usize]>,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
    is_contiguous: bool,
}

impl Shape {
    pub fn reshape<T: AsRef<[usize]>>(&self, shape: T) -> Shape {
        shape.into()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn stride(&self, index: usize) -> usize {
        self.strides[index]
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn transpose<T: AsRef<[usize]>>(&self, axis: T) -> Shape {
        let mut foo = axis.as_ref()
            .iter()
            .cloned()
            .fold(vec![vec![]; 3], |mut acc, i| {
                acc[0].push(self.axis[i]);
                acc[1].push(self.shape[i]);
                acc[2].push(self.strides[i]);
                acc
            })
            .into_iter()
            .map(|x| x.into_boxed_slice())
            .collect::<Vec<Box<[usize]>>>();

        let is_contiguous = foo[0]
            .windows(2)
            .all(|w| w[0] < w[1]);

        Shape {
            strides: foo.pop().unwrap(),
            shape: foo.pop().unwrap(),
            axis: foo.pop().unwrap(),
            is_contiguous,
        }
    }

    pub(super) fn indices(&self) -> Vec<Vec<usize>> {
        self.shape
            .iter()
            .map(|x| 0..*x)
            .fold(vec![], |acc: Vec<Vec<usize>>, range| {
                if acc.len() == 0 {
                    return range.into_iter()
                        .map(|x| vec![x])
                        .collect::<Vec<Vec<usize>>>()
                }

                acc.iter()
                    .flat_map(|vec| range
                        .clone()
                        .into_iter()
                        .map(|x| {
                            let mut cloned = vec.clone();
                            cloned.push(x);
                            cloned
                        }))
                    .collect::<Vec<Vec<usize>>>()
            })
    }

    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous
    }

    pub(crate) fn expand<S: AsRef<[usize]>>(&self, shape: S) -> Shape {
        let shape = shape.as_ref();
        assert!(
            self.shape().len() > shape.len(),
            "the number of sizes provided ({}) must be greater or equal \
            to the number of dimensions in the tensor ({})",
            self.shape().len(),
            shape.len(),
        );

        let iter = self.shape()
            .iter()
            .cloned()
            .rev()
            .chain(vec![1usize; shape.len() - self.shape().len()])
            .zip(shape.iter().cloned().rev())
            .all(|(a, b)| a == b || a == 1 || b == 1);

        0.into()
        todo!()

        Shape {
            axis: todo!(),
            shape: shape.to_vec().into_boxed_slice(),
            strides: todo!(),
            is_contiguous: false,
        }
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape {
            axis: [].into(),
            shape: [].into(),
            strides: [].into(),
            is_contiguous: true,
        }
    }
}

impl<S: AsRef<[usize]>> From<S> for Shape {
    fn from(shape: S) -> Self {
        let items_count = shape
            .as_ref()
            .iter()
            .product::<usize>();

        Shape {
            axis: (0..shape.as_ref().len())
                .collect::<Vec<usize>>()
                .into_boxed_slice(),
            shape: shape.as_ref().into(),
            strides: shape.as_ref()
                .iter()
                .scan(items_count, |items_count, dim_size| {
                    *items_count = *items_count / *dim_size;
                    (*items_count).into()
                })
                .collect(),
            is_contiguous: true,
        }
    }
}
