import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout)

def build_model():
    def eye_processing_layers():
        eye_input = Input(shape=(64, 64, 3))
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(eye_input)
        conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)
        droput = Dropout(0.25)(max_pooling)
        flatten = Flatten()(droput)
        dense = Dense(16)(flatten)
        return eye_input, dense

    left_eye_input, left_eye_output = eye_processing_layers()
    right_eye_input, right_eye_output = eye_processing_layers()

    face_points_input = Input(shape=(72, 2))
    face_points_flatten = Flatten()(face_points_input)
    face_points_dense1 = Dense(32)(face_points_flatten)
    face_points_dense2 = Dense(24)(face_points_dense1)

    all_data = keras.layers.concatenate(
            [face_points_dense2, left_eye_output, right_eye_output])
    all_data_dense1 = Dense(32)(all_data)
    all_data_dense2 = Dense(16)(all_data_dense1)
    #output = Dense(2, activation='linear')(all_data_dense2)
    output = Dense(2)(all_data_dense2)

    model = Model(inputs=[left_eye_input, right_eye_input, face_points_input],
            outputs=[output])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'])

    return model
