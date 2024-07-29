from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPooling2D, Lambda
import tensorflow.keras.backend as K

def build_siamese_model(input_shape, num_classes):
    # 输入层
    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)

    # 共享的卷积层
    convolution_layer = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')
    
    # 左侧图像处理
    # 第一层卷积
    x_left = convolution_layer(input_left)
    x_left = MaxPooling2D(pool_size=(2, 2))(x_left)
    x_left = Dropout(0.25)(x_left)
    
    convolution_layer = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')
    # 第二层卷积
    x_left = convolution_layer(x_left)
    x_left = MaxPooling2D(pool_size=(2, 2))(x_left)
    x_left = Dropout(0.25)(x_left)
    
    convolution_layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')
    # 第三层卷积
    x_left = convolution_layer(x_left)
    x_left = MaxPooling2D(pool_size=(2, 2))(x_left)
    x_left = Dropout(0.25)(x_left)

    # 展平层
    x_left = Flatten()(x_left)

    # 共享的卷积层
    convolution_layer = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')
    
    # 右侧图像处理
    x_right = convolution_layer(input_right)
    x_right = MaxPooling2D(pool_size=(2, 2))(x_right)
    x_right = Dropout(0.25)(x_right)
    convolution_layer = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')
    x_right = convolution_layer(x_right)
    x_right = MaxPooling2D(pool_size=(2, 2))(x_right)
    x_right = Dropout(0.25)(x_right)
    convolution_layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')
    x_right = convolution_layer(x_right)
    x_right = MaxPooling2D(pool_size=(2, 2))(x_right)
    x_right = Dropout(0.25)(x_right)
    x_right = Flatten()(x_right)

    # 计算两个图像特征的差异
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([x_left, x_right])
    
    # 密集连接层
    x = Dense(512, activation='relu')(L1_distance)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    # 创建模型
    model = Model(inputs=[input_left, input_right], outputs=output_layer)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
