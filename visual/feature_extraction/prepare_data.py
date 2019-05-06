import numpy as np

training_visual_features_extract = np.load('extracts_training_set/visual_features_training_extract.npy')
training_gaze_features_extract = np.load('extracts_training_set/gaze_features_training_extract.npy')
training_pose_features_extract = np.load('extracts_training_set/pose_features_training_extract.npy')
training_action_units_extract = np.load('extracts_training_set/action_units_training_extract.npy')

np.save('CLNF_training_set_combined_features.npy', np.hstack((training_visual_features_extract, training_gaze_features_extract, training_pose_features_extract, training_action_units_extract)))

dev_visual_features_extract = np.load('extracts_development_set/visual_features_development_extract.npy')
dev_gaze_features_extract = np.load('extracts_development_set/gaze_features_development_extract.npy')
dev_pose_features_extract = np.load('extracts_development_set/pose_features_development_extract.npy')
dev_action_units_extract = np.load('extracts_development_set/action_units_development_extract.npy')

np.save('CLNF_development_set_combined_features.npy', np.hstack((dev_visual_features_extract, dev_gaze_features_extract, dev_pose_features_extract, dev_action_units_extract)))