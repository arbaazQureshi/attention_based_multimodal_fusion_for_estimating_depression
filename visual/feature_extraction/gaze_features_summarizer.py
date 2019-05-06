from keras.models import load_model, Model
from load_model.load_gaze_features import load_training_data, load_development_data
import numpy as np
import os

os.environ['VISIBLE_CUDA_DEVICES'] = '0'

gaze_features_model = load_model('best_solo_models/gaze_features.h5')
gaze_features_extractor = Model(inputs = gaze_features_model.inputs, outputs = gaze_features_model.layers[1].output)

X_train, Y_train, X_train_gender = load_training_data()
X_dev, Y_dev, X_dev_gender = load_development_data()

X_train_extract = gaze_features_extractor.predict([X_train, X_train_gender], batch_size = X_train.shape[0])
np.save('training_set_extracts/gaze_features_training_extract.npy', X_train_extract)

X_dev_extract = gaze_features_extractor.predict([X_dev, X_dev_gender], batch_size = X_dev.shape[0])
np.save('development_set_extracts/gaze_features_development_extract.npy', X_dev_extract)