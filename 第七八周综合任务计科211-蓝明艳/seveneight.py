# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/digit-recognizer/sample_submission.csv
/kaggle/input/digit-recognizer/train.csv
/kaggle/input/digit-recognizer/test.csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

import os
import PIL

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# hide warnings
import warnings
warnings.filterwarnings('ignore')

# set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')
df.head()
df.shape
y =df['label']

X=df.drop(columns = 'label')
X = np.array(X, dtype="float32")
X = X/255.0
img_height, img_width = 28,28

X = X.reshape(-1, img_height, img_width, 1)

y = to_categorical(y, num_classes=10)
seed =100
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.33 ,random_state = seed)
LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3,mode='min', epsilon=0.0001, cooldown=0, min_lr=0.00001)
ES = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
callbacks_list = [LR,ES]
# Buildinf a sequential model with  rescaling and 3 conv layers and 2 dense layers with softmax as output activation
model = Sequential([
    # 1st conv layer
  layers.Conv2D(32, 3, padding='same', activation='relu',input_shape=(img_height,img_width,1)),
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
    # maxpooling layer
  layers.MaxPooling2D(),
    # 2nd conv layer
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
    # maxpooling layer
  layers.MaxPooling2D(),
    # 3rd conv layer
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
     # maxpooling layer
  layers.MaxPooling2D(),
     # flatten
  layers.Flatten(),
  layers.Dropout(0.25),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.25),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.25),
  layers.Dense(10, activation='softmax')
])
### Compiling the model using adam optimiser and categorical_crossentropy loss function
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# training  the model with 50 epochs
epochs = 50
history = model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),
  steps_per_epoch = X_train.shape[0]/32,
  callbacks = callbacks_list,
  epochs=epochs
)
df_test.head()
df_test.shape
test = np.array(df_test, dtype=np.float32)/255
test = test.reshape(-1,28,28,1)
prediction = model.predict(test)
predict = np.array(np.round(prediction), dtype = np.int32)
predict = np.argmax(predict , axis=1).reshape(-1, 1)
submission_df = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
submission_df['Label'] = predict
submission_df.to_csv('submission.csv', index=False)

