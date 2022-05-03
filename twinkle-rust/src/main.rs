mod tensor;

use crate::tensor::index::Index;
use crate::tensor::tensor::Tensor;

fn load_digits() -> (Tensor<f32>, Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let x_train = Tensor::from(vec![0f32; 1000 * 8 * 8])
        .reshape(&[1000, 64]);
    let y_train = Tensor::from(vec![0f32; 1000 * 10])
        .reshape(&[1000, 10]);

    let x_test = Tensor::from(vec![0f32; 767 * 8 * 8])
        .reshape(&[767, 64]);
    let y_test = Tensor::from(vec![0f32; 767 * 10])
        .reshape(&[767, 10]);

    (x_train, x_test, y_train, y_test)
}

fn main() {
    let foo: Tensor<i32> = (0..24).collect::<Vec<i32>>().into();

    let foo = foo.reshape(&[2, 3, 2, 2]);
    let bar = foo.transpose(&[1, 0, 3, 2]);

    println!("{:?}", bar.index(&[1, 0, 1, 0]));

    println!("shape {:?} strides {:?}", foo.shape(), foo.strides());
    println!("shape {:?} strides {:?}", bar.shape(), bar.strides());

    for item in bar {
        println!("{:?}", item);
    }

    let (x_train, x_test, y_train, y_test) = load_digits();

    let lr: Tensor<f32> = 0.01.into();
    let mut w = Tensor::from(vec![0f32; 64 * 10])
        .reshape(&[64, 10]);

    let mut correct_answers_count = 0u32;

    let tensor: Tensor<i32> = Tensor::<i32>::from((0..12).collect::<Vec<i32>>())
        .reshape(&[3, 4, 1]);
    tensor.expand(&[2, 2, 3, 4, 2]);

    for epoch in 0..2 {
        for i in 0..x_train.len() {
            let x_i = x_train.index(&[i]).reshape(&[1, 64]);
            let y_i = y_train.index(&[i]).reshape(&[1, 10]);

            let y_hat = x_i.mm(&w);
            correct_answers_count += u32::from(y_hat.argmax().item() == y_i.argmax().item());

            let loss = F::mse_loss(&y_hat, &y_i, "mean");
            let y_delta = &y_hat - &y_i;
            let loss_grad = x_i.T().mm(&y_delta);
            println!("loss {}", loss.item());

            // w -= &lr * &loss_grad;
            // loss.backward();
        }
    }
}
