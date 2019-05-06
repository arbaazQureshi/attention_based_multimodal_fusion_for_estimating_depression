import numpy as np

training_COVAREP_output_encoding = np.load('training_set_encoding/COVAREP_output_training_encoding.npy')
training_FORMANT_features_encoding = np.load('training_set_encoding/FORMANT_features_training_encoding.npy')

np.save('speech_training_set_combined_features.npy', np.hstack((training_COVAREP_output_encoding, training_FORMANT_features_encoding)))



dev_COVAREP_output_encoding = np.load('development_set_encoding/COVAREP_output_development_encoding.npy')
dev_FORMANT_features_encoding = np.load('development_set_encoding/FORMANT_features_development_encoding.npy')

np.save('speech_development_set_combined_features.npy', np.hstack((dev_COVAREP_output_encoding, dev_FORMANT_features_encoding)))