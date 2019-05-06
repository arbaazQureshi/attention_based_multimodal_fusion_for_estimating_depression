from keras.models import load_model, Model
from load_model.load_FORMANT_features import load_training_data, load_development_data
import numpy as np
import os

os.environ['VISIBLE_CUDA_DEVICES'] = '0'

FORMANT_features_model = load_model('best_solo_models/FORMANT.h5')
FORMANT_features_extractor = Model(inputs = FORMANT_features_model.inputs, outputs = FORMANT_features_model.layers[5].output)

X_train, Y_train, X_train_gender = load_training_data()
X_dev, Y_dev, X_dev_gender = load_development_data()

X_train_encoding = FORMANT_features_extractor.predict([X_train, X_train_gender], batch_size = X_train.shape[0])
np.save('training_set_encoding/FORMANT_features_training_encoding.npy', X_train_encoding)

X_dev_encoding = FORMANT_features_extractor.predict([X_dev, X_dev_gender], batch_size = X_dev.shape[0])
np.save('development_set_encoding/FORMANT_features_development_encoding.npy', X_dev_encoding)