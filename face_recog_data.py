# format_data.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.utils.io_utils import load_pickle,save_pickle


N=23700 # Clips off 5 samples (at random) to make total divisible by 150,
# the desired batch size

savename_features = '/media/yuhongluo/face_recog/features.pkl'
savename_labels = '/media/yuhongluo/face_recog/labels.pkl'
savename_sensitive_attrs = '/media/yuhongluo/face_recog/sensitive_attrs.pkl'

print("loading data...")
data = pd.read_csv('/media/yuhongluo/face_recog/age_gender.csv')
# Shuffle data since it is in order of age, then gender
data = data.sample(n=len(data),random_state=42).iloc[:N]
# Convert pixels from string to numpy array
print("Converting pixels to array...")
data['pixels']=data['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))

# normalize pixels data
print("Normalizing and reshaping pixel data...")
data['pixels'] = data['pixels'].apply(lambda x: x/255)

# Reshape pixels array
X = np.array(data['pixels'].tolist())

## Converting pixels from 1D to 4D
features = X.reshape(X.shape[0],1,48,48)

# Extract gender labels
labels = data['gender'].values

# Make one-hot sensitive feature columns
M=data['gender'].values
mask=~(M.astype("bool"))
F=mask.astype('int64')
sensitive_attrs = np.hstack((M.reshape(-1,1),F.reshape(-1,1)))

# Save to pickle files
print("Saving features, labels, and sensitive_attrs to pickle files")
save_pickle(savename_features,features)
save_pickle(savename_labels,labels)
save_pickle(savename_sensitive_attrs,sensitive_attrs)