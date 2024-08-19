import cvnn.layers as complex_layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def build_CVNNs_model(input_shape, num_classes):
    inputs = complex_layers.complex_input(shape=input_shape)

    # 第一层卷积
    x = complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu')(inputs)
    x = complex_layers.ComplexMaxPooling2D((2, 2))(x)
    x = complex_layers.ComplexDropout(0.25)(x)  # 添加丢弃层

    # # 第二层卷积
    # x = complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu')(x)
    # x = complex_layers.ComplexMaxPooling2D((2, 2))(x)
    # x = complex_layers.ComplexDropout(0.25)(x)  # 添加丢弃层
    #
    # # 第三层卷积
    # x = complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu')(x)
    # x = complex_layers.ComplexMaxPooling2D((2, 2))(x)
    # x = complex_layers.ComplexDropout(0.25)(x)  # 添加丢弃层

    # 展平层
    x = complex_layers.ComplexFlatten()(x)

    # 复值密集连接层
    x = complex_layers.ComplexDense(64, activation='cart_relu')(x)
    x = complex_layers.ComplexDropout(0.5)(x)

    # 输出层
    output_layer = complex_layers.ComplexDense(num_classes, activation='softmax_real_with_abs')(x)

    # 构建模型
    model = tf.keras.Model(inputs, output_layer)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
