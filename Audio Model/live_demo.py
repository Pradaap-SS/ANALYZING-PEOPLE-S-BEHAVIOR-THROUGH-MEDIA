#LIVE DEMO
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

import os
import pandas as pd
import librosa
import glob 

import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys

from keras.models import load_model

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

loaded_model= load_model('saved_models/Emotion_Voice_Detection_Model.h5')
print("Loaded model from disk")

loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")

lb = LabelEncoder()


data, sampling_rate = librosa.load('happy.wav')

%pylab inline
import os
import pandas as pd
import librosa
import glob 


plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('happy.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive

livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
livedf2

twodim= np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)




livepreds1=livepreds.argmax(axis=1)
print(livepreds1)
liveabc = livepreds1.astype(int).flatten()
print(liveabc)

livepredictions = (lb.inverse_transform(livepreds1))
print(livepredictions)
