# from tensorflow.keras.layers import Input, Conv3D, AveragePooling2D, Activation, Dense, Flatten
# from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv3D, AveragePooling3D, MaxPool3D,  Activation, Dense, Flatten
from tensorflow.keras.models import Model



##### from rust, the orders are as following:
#(2D) input_dims(): (bath_size, in_channels, in_height, in_width)
#(2D) kernel.dim(): (num, k_channels, k_height, k_width)

#(3D) input_dims(): (bath_size, in_channels, in_depth, in_height, in_width)
#(3D) kernel.dim(): (num, k_channels, k_depth, k_height, k_width)


### from the python (2D), since it use tensorflow library
### the library are used as input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".


### model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
### model.add(MaxPooling2D(pool_size=(2, 2)))

### Here we are learning a total of 32 filters 
### and then we use Max Pooling to reduce the spatial dimensions of the output volume.
## This parameter determines the dimensions of the kernel. Common dimensions include 1×1, 3×3, 5×5, and 7×7 which can be passed as (1, 1), (3, 3), (5, 5), or (7, 7) tuples.
## It is an integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
## This parameter must be an odd integer.



##### python (3D) The inputs are 28x28x28 volumes with a single channel, and the
# batch size is 4
##input_shape =(4, 28, 28, 28, 1)


###tf.keras.layers.Conv3D(
###    filters, kernel_size, strides=(1, 1, 1), padding='valid', ....)


def build():
    input_shape = (16, 112, 112, 112, 3)
    img_input = Input(shape=input_shape)

    x = img_input

    #linear 1
    # Conv3D ((number of filters), (dimensions of kernel), stides, padding ,....)
    x = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)

    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)



    #linear 2
    x = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)


    #linear 3
    x = Conv3D(256, (3, 3, 3),strides=(2, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)


    #linear 4
    x = Conv3D(256, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)



    #linear 5
    x = Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)
  

     #linear 6
    x = Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    
    #linear 7

    x = Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)
  

    #linear 8
    x = Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Activation('approx_activation')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    #padding=(1, 1, 1)
    x = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    x = Flatten()(x)

    x = Dense(4096)(x)
    x = Activation('approx_activation')(x)
    x = Dropout(.5)(x)

    x = Dense(4096)(x)
    x = Activation('approx_activation')(x)
    x = Dropout(.5)(x)


    x = Dense(487)(x)
    #model.add(Dense(487, activation='softmax', name='fc8'))
    x = Activation('approx_activation')(x)

     #x = Flatten()(x)
    #x = Dense(101, activation='softmax')(x)
    return Model(img_input, x)




    # 
    # x = Conv3D(16, (1, 1, 1),
    #                         strides=(1, 1),
    #                         padding='valid')(x)
    # x = Activation('approx_activation')(x)
    

    ###### Otheres #########

    # model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
    #                        border_mode='valid', name='pool5'))
    # model.add(Flatten())
    # # FC layers group
    # model.add(Dense(4096, activation='relu', name='fc6'))
    # model.add(Dropout(.5))
    # model.add(Dense(4096, activation='relu', name='fc7'))
    # model.add(Dropout(.5))
    # model.add(Dense(487, activation='softmax', name='fc8'))




  