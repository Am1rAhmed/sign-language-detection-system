import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,multilabel_confusion_matrix,classification_report)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from load_dataset import load_dataset

x, y, actions = load_dataset()


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
stratify=np.argmax(y, axis=1),
random_state=42
)

model = load_model("sign_language_model.h5")

yhat = model.predict(x_test, verbose=0)

ytrue = np.argmax(y_test,axis=1)

ypred = np.argmax(yhat,axis=1)

print("Accuracy:",accuracy_score(ytrue,ypred))

print(multilabel_confusion_matrix(ytrue,ypred))

print(classification_report(ytrue,ypred,target_names=actions))
