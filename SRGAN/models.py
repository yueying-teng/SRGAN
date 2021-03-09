import tensorflow as tf
from tensorflow.python.keras.layers import Add, Dropout, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19


def normalize_01(x):
    """
    Normalizes RGB images to [0, 1].
    """
    return x / 255.0


def denormalize_01(x):
    """
    Inverse of normalize_01
    """
    return x * 255.0


def normalize_m11(x):
    """
    Normalizes RGB images to [-1, 1].
    """
    return x / 127.5 - 1


def denormalize_m11(x):
    """
    Inverse of normalize_m11.
    """
    return (x + 1) * 127.5


##### build EDSR ####

def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = edsr_res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = edsr_upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize_01)(x)
    return Model(x_in, x, name="edsr")


def edsr_generator(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    """
    has input normalized to [-1, 1]
    and uses tanh as the activation function for the last layer
    """
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_m11)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = edsr_res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = edsr_upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same', activation='tanh')(x)

    x = Lambda(denormalize_m11)(x)
    return Model(x_in, x, name="edsr")


def edsr_res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def edsr_upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(subpixel_conv2d(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


##### build SRGAN ####
# https://github.com/krasserm/super-resolution/

def upsample(x_in, num_filters):
    # Subpixel Conv will upsample from (h, w, c) to (hr, wr, c/r^2)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(subpixel_conv2d(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


# def upsample(x_in, num_filters):
#     x = UpSampling2D()(x_in)
#     x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
#     x = BatchNormalization()(x)
#     return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4)
    x = upsample(x, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(first_dim, sec_dim, num_filters=64, add_dropout=False):
    x_in = Input(shape=(first_dim, sec_dim, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)
    x = Dense(1024)(x)
    if add_dropout:
        x = Dropout(0.4)(x)
    x = LeakyReLU(alpha=0.2)(x)
    # https://github.com/soumith/ganhacks/issues/36#issuecomment-492964089
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


def sr_resnet_simple(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters)
    x = upsample(x, num_filters)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)


def discriminator_simple(first_dim, sec_dim, num_filters=64):
    x_in = Input(shape=(first_dim, sec_dim, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    # x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    # x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    # x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    # x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    # https://github.com/soumith/ganhacks/issues/36#issuecomment-492964089
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


def vgg_19():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    for l in vgg.layers: 
        l.trainable = False
        
    return Model(vgg.input, vgg.layers[output_layer].output)


