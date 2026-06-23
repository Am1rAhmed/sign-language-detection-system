import numpy as np
import os
from tensorflow.keras.utils import to_categorical


DATA_PATH = os.path.join("MP_Data")

actions = np.array([
'Hello',
'Thanks',
'I love you',
'Help',
'Sorry',
'Eat',
'Nice to meet you'
])

no_sequences = 85
TARGET_LENGTH = 65

def load_dataset():

    label_map = {
    label: num
    for num, label in enumerate(actions)
    }

    sequences = []
    labels = []

    for action in actions:

        for sequence in range(no_sequences):
            window = []

            sequence_path = os.path.join(
                DATA_PATH,
                action,
                str(sequence)
            )

            files = sorted(
                os.listdir(sequence_path),
                key=lambda x: int(x.split('.')[0])
            )

            for file in files:

                window.append(
                    np.load(
                        os.path.join(
                            sequence_path,
                            file
                        )
                    )
                )

            while len(window) < TARGET_LENGTH:
                window.append(window[-1])

            window = window[:TARGET_LENGTH]

            sequences.append(window)
            labels.append(label_map[action])


    x = np.array(sequences,dtype=np.float32)

    y = to_categorical(labels).astype(np.float32)

    return x, y, actions