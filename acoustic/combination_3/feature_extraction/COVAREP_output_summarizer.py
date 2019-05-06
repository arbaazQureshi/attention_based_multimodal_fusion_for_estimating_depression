from keras.models import load_model, Model
from load_model.load_COVAREP_output import load_training_data, load_development_data
import numpy as np
import os

os.environ['VISIBLE_CUDA_DEVICES'] = '0'

COVAREP_model = load_model('best_solo_models/COVAREP.h5')
COVAREP_extractor = Model(inputs = COVAREP_model.inputs, outputs = COVAREP_model.layers[1].output)

X_train, Y_train, X_train_gender = load_training_data()
X_dev, Y_dev, X_dev_gender = load_development_data()

X_train_encoding = COVAREP_extractor.predict([X_train, X_train_gender], batch_size = int(X_train.shape[0]/5)+1)
np.save('training_set_encoding/COVAREP_output_training_encoding.npy', X_train_encoding)

X_dev_encoding = COVAREP_extractor.predict([X_dev, X_dev_gender], batch_size = int(X_dev.shape[0]/5)+1)
np.save('development_set_encoding/COVAREP_output_development_encoding.npy', X_dev_encoding)