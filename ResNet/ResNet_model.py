from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Add, Conv2D, Input, AveragePooling2D, Flatten, Dense, Activation
from tensorflow.keras.models import Model


def residual_block(x, num_channels, strides=(1, 1)):
    shortcut = x
    x = Conv2D(num_channels, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(x)
    x = Activation("relu")(x)
    x = Conv2D(num_channels, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    if shortcut.shape[1:] != x.shape[1:]:
        shortcut = Conv2D(num_channels, kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)(shortcut)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x


# 每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
def resnet_block(x, num_channels, strides):
    x = residual_block(x, num_channels, strides=strides)
    x = residual_block(x, num_channels)
    return x


def ResNet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(inputs)
    x = Activation("relu")(x)

    x = resnet_block(x, 32, strides=(1, 1))
    x = resnet_block(x, 64, strides=(2, 2))
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = resnet_block(x, 128, strides=(2, 2))
    x = resnet_block(x, 256, strides=(2, 2))
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def LinearClassifier(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    encoder = ResNet(input_shape)
    feature = encoder(inputs)
    logits = Dense(num_classes, name='logits')(feature)
    output_layer = Activation("softmax")(logits)
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model = Model(inputs=inputs, outputs=output_layer)
    # 编译模型
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
