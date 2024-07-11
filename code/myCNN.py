from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPooling2D


def build_cnn_model(input_shape, num_classes):
    # 输入层
    input_layer = Input(shape=input_shape)

    # 第一层卷积
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # 第二层卷积
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # 第三层卷积
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # 展平层
    x = Flatten()(x)

    # 密集连接层
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    # 创建模型
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
