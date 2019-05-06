from keras.models import load_model, Model
from load_model.load_visual_features import load_training_data, load_development_data
import numpy as np
import os

os.environ['VISIBLE_CUDA_DEVICES'] = '0'

visual_features_model = load_model('best_solo_models/visual_features.h5')
visual_features_extractor = Model(inputs = visual_features_model.inputs, outputs = visual_features_model.layers[1].output)

X_train, Y_train, X_train_gender = load_training_data()
X_dev, Y_dev, X_dev_gender = load_development_data()

X_train_extract = visual_features_extractor.predict(X_train, batch_size = 5)
np.save('training_set_extracts/visual_features_training_extract.npy', X_train_extract)

X_dev_extract = visual_features_extractor.predict(X_dev, batch_size = X_dev.shape[0])
np.save('development_set_extracts/visual_features_development_extract.npy', X_dev_extract)
