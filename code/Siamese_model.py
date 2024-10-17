from cvnn.layers import complex_input, ComplexConv2D, ComplexFlatten, ComplexDense, ComplexMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf

# 构建共享网络
def build_shared_model(input_shape):
    inputs = complex_input(shape=input_shape)
    x = ComplexConv2D(32, kernel_size=(3, 3), activation='cart_relu', padding='same')(inputs)
    x = ComplexMaxPooling2D(pool_size=(2, 2))(x)

    x = ComplexConv2D(64, kernel_size=(3, 3), activation='cart_relu', padding='same')(x)
    x = ComplexMaxPooling2D(pool_size=(2, 2))(x)

    x = ComplexConv2D(128, kernel_size=(3, 3), activation='cart_relu', padding='same')(x)
    x = ComplexMaxPooling2D(pool_size=(2, 2))(x)

    x = ComplexConv2D(256, kernel_size=(3, 3), activation='cart_relu', padding='same')(x)
    x = ComplexMaxPooling2D(pool_size=(2, 2))(x)

    x = ComplexFlatten()(x)

    x = ComplexDense(512, activation='cart_relu')(x)

    x = ComplexDense(256, activation='cart_relu')(x)

    x = ComplexDense(128, activation='cart_relu')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def contrastive_loss(y_true, y_pred):
    margin = 0.1

    square_pred = y_pred

    margin_square = K.maximum(margin - y_pred, 0)

    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def build_siamese_model(input_shape):
    shared_model = build_shared_model(input_shape)
    # 输入层
    input_sample = complex_input(shape=input_shape)
    input_average = complex_input(shape=input_shape)

    # 全连接层
    x_sample = shared_model(input_sample)
    x_average = shared_model(input_average)

    x_sample =tf.abs(x_sample)
    x_average = tf.abs(x_average)
    l1_distance = Lambda(
        lambda tensors: K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True))(
        [x_sample, x_average])

    # 创建模型
    model = Model([input_sample, input_average], l1_distance)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=contrastive_loss, metrics=[])

    return model
