use crate::tensor::*;

fn main() {
    let data = (0..100).collect::<Vec<i32>>();
    let foo: Tensor<i32> = Tensor::<i32>::from(&data[..])
        .reshape(&[2, 5, 10][..]);

    println!("{:?}", foo.strides());
    println!("{:?}", foo.stride(0));
    println!("{:?}", foo.stride(2));
}
