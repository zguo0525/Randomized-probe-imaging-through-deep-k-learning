# tensorflow tools used in "Randomized probe imaging through deep k-learning"
# written and maintained by Abe Levitan and Zhen Guo
# =============================================================================
"""Contains tools to generate deep k-learning architecture."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Concatenate, Conv2DTranspose, Dropout, LeakyReLU, ZeroPadding2D, Input
from tensorflow.keras import Model


def norm_to_255(tensor):
    """normalize data to range between 0 and 255"""
    tf_max = tf.math.reduce_max(tensor)
    tf_min = tf.math.reduce_min(tensor)
    return 255 * (tensor - tf_min) / (tf_max - tf_min)


def DoubleConv(filters, kernel_size, initializer='glorot_uniform'):
    """Double Convolution layers"""
    def layer(x):

        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    return layer


def Conv2DTranspose_block(filters, kernel_size=(3, 3), transpose_kernel_size=(2, 2), upsample_rate=(2, 2),
                          initializer='glorot_uniform', skip=None):
    """Conv2DTranspose block"""
    def layer(input_tensor):

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same')(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = DoubleConv(filters, kernel_size, initializer=initializer)(x)

        return x

    return layer


def _get_efficient_unet(encoder, out_channels=2, dropout=0.1, concat_input=True):
    """Define efficient U-net"""
    
    # name of the MBconv layers in the efficientnetB7
    skip_candidates = ['1a', '2a', '3a', '4a']

    # MBConvBlocks has outputs from the efficientB07 from 1a to 4a layers,
    # the blocks will be used in the process of concatenation
    # to get a U-net like topology for the efficientUnet
    MBConvBlocks = []
    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('block' + mbblock_nr + '_project_bn').output
        MBConvBlocks.append(mbblock)

    # We used Conv2DTranspose as the upsampling block
    UpBlock = Conv2DTranspose_block
    
    # get the last output of the efficientB07 before the classification head
    o = encoder.output
    
    # four upsampling modules with four concatenations
    for filters in [512, 256, 128, 64]:
        o = UpBlock(filters, skip=MBConvBlocks.pop())(o)
        o = Dropout(dropout)(o)   
        
    if concat_input:
        o = UpBlock(32, skip=encoder.input)(o)
    else:
        o = UpBlock(32, skip=None)(o)
    # last layer of Conv2D to set the channel number to be 1
    o = Conv2D(out_channels, (1, 1), padding='same')(o)

    model = Model(encoder.input, o)
    
    return model


def get_efficient_unet_b7(input_shape, out_channels=2, dropout=0.1, pretrained=True, concat_input=True):
    """Get a Unet model with Efficient-B7 encoder
    :param input_shape: shape of input (cannot have None element)
    :param out_channels: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B7 model
    """
    if pretrained is True:
        encoder = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=None,
                                                        input_shape=input_shape, pooling=None, classes=1000,
                                                        classifier_activation='softmax')
        # freeze the encoder part if it's loaded with pre-trained weights
        encoder.trainable = False
    else:
        encoder = tf.keras.applications.EfficientNetB7(include_top=False, weights=None, input_tensor=None,
                                                        input_shape=input_shape, pooling=None, classes=1000,
                                                        classifier_activation='softmax')
        encoder.trainable = True

    model = _get_efficient_unet(encoder, out_channels, 
                                dropout=dropout,concat_input=concat_input)
    return model


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    """discriminator loss for GAN using BinaryCrossentropy
    """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def downsample(filters, size, apply_batchnorm=True):
    """define downsample block for Discriminator model"""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result


def Discriminator():
    """define Discriminator model"""
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = Input(shape=[256, 256, 1], name='input_image')
    tar = Input(shape=[256, 256, 1], name='target_image')

    x = Concatenate()([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(batchnorm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return Model(inputs=[inp, tar], outputs=last)

def npcc(truth, guess):
    """Compare a guess image and a true image using npcc
    """
    fsp = guess - tf.reduce_mean(guess)
    fst = truth - tf.reduce_mean(truth)

    devP = tf.math.reduce_std(guess)
    devT = tf.math.reduce_std(truth)

    # npcc = cov(truth, guess)/[std(truth)*std(guess)]
    # we clip the value with minimal of to avoid dividing zero
    loss_pcc = (-1) * tf.reduce_mean(fsp * fst) / tf.clip_by_value(devP * devT, 1e-7, 1e12)    

    return loss_pcc
