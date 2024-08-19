import cvnn.layers as complex_layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda


def build_CVNNs_model(input_shape, num_classes):
    # 输入层
    input_left = complex_layers.complex_input(shape=input_shape)
    input_right = complex_layers.complex_input(shape=input_shape)

    # 左侧处理
    # 第一层卷积
    x_left = complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu')(input_left)
    x_left = complex_layers.ComplexMaxPooling2D((2, 2))(x_left)
    x_left = complex_layers.ComplexDropout(0.25)(x_left)  # 添加丢弃层

    # 展平层
    x_left = complex_layers.ComplexFlatten()(x_left)

    # 右侧处理
    # 第一层卷积
    x_right = complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu')(input_right)
    x_right = complex_layers.ComplexMaxPooling2D((2, 2))(x_right)
    x_right = complex_layers.ComplexDropout(0.25)(x_right)  # 添加丢弃层

    # 展平层
    x_right = complex_layers.ComplexFlatten()(x_right)

    # 计算两个图像特征的差异
    l1_distance = Lambda(lambda tensors: tf.subtract(tensors[0], tensors[1]))([x_left, x_right])

    # 复值密集连接层
    x = complex_layers.ComplexDense(64, activation='cart_relu')(l1_distance)
    x = complex_layers.ComplexDropout(0.5)(x)

    # 输出层
    output_layer = complex_layers.ComplexDense(num_classes, activation='softmax_real_with_abs')(x)

    # 构建模型
    model = tf.keras.Model([input_left, input_right], output_layer)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
