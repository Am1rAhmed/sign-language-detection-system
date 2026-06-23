import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from load_dataset import load_dataset

x, y, actions = load_dataset()

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=np.argmax(y, axis=1),
    random_state=42
)


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()

model.add(LSTM(128,
               return_sequences=True,
               input_shape=(65,258)))

model.add(LSTM(256,
               return_sequences=True))

model.add(LSTM(128,
               return_sequences=False))

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(actions.shape[0],
                activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=70, callbacks=[tb_callback])

model.save("sign_language_model.h5")
print("Model saved successfully")