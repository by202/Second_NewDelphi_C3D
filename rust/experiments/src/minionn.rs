use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_minionn<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        0 => vec![1, 3, 6, 8, 11, 13, 15],
        1 => vec![1, 3, 6, 8, 11, 13],
        2 => vec![1, 3, 6, 8, 11],
        3 => vec![3, 11, 13, 15],
        5 => vec![6, 11],
        6 => vec![11],
        7 => vec![],
        _ => unreachable!(),
    };

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers:      vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.


    // (2d): (batch_size, channel, h, w)

    // (3d): (batch_size, channel, d, h, w)

    let input_dims = (batch_size, 3, 774, 112, 112);


    // https://github.com/aslucki/C3D_Sport1M_keras/blob/master/c3d/sport1m_model.py

    // let input_shape = (16, 3, 112, 112)


    // from linear_only.rs file
    //sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;

    // 1
    //window size 3*3; stride(1,1); pad(1,1);
    // number of output channels 64
    let kernel_dims = (64, 3, 3, 3, 3);
    // sample_conv_layer(vs, input_dims,kernel_dims, stride,padding,rng)
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    //relu activation
    add_activation_layer(&mut network, &relu_layers);

    // 2 conv windows size 3*3, stide(1,1), pad(1,1), the number of output channels 64
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    //relu activation
    add_activation_layer(&mut network, &relu_layers);

    // 3
    // mean pooling window size
    // window size 1*2*2, output 
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2, 2), 2);
    network.layers.push(Layer::LL(pool));

    // 4 window size 3*3, stide (1,1) and pad (1,1)
    // the number of output channels 64
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (128, 64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // relu activation
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    // 5 convolu window 3*3, stide (1,1), pad(1,1) 
    // number of output 64
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 128, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // Relu activation
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    // 6 mean pooling: window size 3*3, stride (1,1), pad (1,1)
    // number of output channels 64
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2, 2), 2);
    network.layers.push(Layer::LL(pool));


    // 7 conv window 3*3, stide(1,1),pad(1,1)
    // number of output channel 64
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 256, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    //  conv window size 1*1, stide (1,1), number of channel 64
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 256, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    // 9conv  widnows 1*1, stride (1,1), number of channel  16
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 256, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    // 10
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 101, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());

    network
}
