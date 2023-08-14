# format_data.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.utils.io_utils import load_pickle,save_pickle

savename_features = '/media/yuhongluo/health/features.pkl'
savename_mortal_labels = '/media/yuhongluo/health/mortal_labels.pkl'
savename_gender_labels = '/media/yuhongluo/health/gender_labels.pkl'
savename_sensitive_attrs = '/media/yuhongluo/health/sensitive_attrs.pkl'

print("loading data...")
d = pd.read_csv('./static/datasets/supervised/health_vfae/health.csv')
# Shuffle data
d = d.sample(n=len(d),random_state=42)

d = d[d['YEAR_t'] == 'Y3']
sex = d['sexMISS'] == 0
age = d['age_MISS'] == 0
d = d.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
d = d[sex & age]



def gather_labels(df):
    labels = []
    for j in range(df.shape[1]):
        if type(df[0, j]) is str:
            labels.append(np.unique(df[:, j]).tolist())
        else:
            labels.append(np.median(df[:, j]))
    return labels

ages = d[['age_%d5' % (i) for i in range(0, 9)]]
sexs = d[['sexMALE', 'sexFEMALE']]
charlson = d['CharlsonIndexI_max']
#   ['age_%d5' % (i) for i in range(0, 9)] + 'sexMALE', 'sexFEMALE',
x = d.drop(
    ['CharlsonIndexI_max', 'CharlsonIndexI_min',
                                              'CharlsonIndexI_ave', 'CharlsonIndexI_range', 'CharlsonIndexI_stdev',
                                              'trainset'], axis=1).values

labels = gather_labels(x)
xs = np.zeros_like(x)
for i in range(len(labels)):
    xs[:, i] = x[:, i] > labels[i]
x = xs[:, np.nonzero(np.mean(xs, axis=0) > 0.05)[0]].astype(np.float32)

charlson_labels = (charlson.values > 0).astype(np.int64)
gender_labels = sexs.values[:, 1]
# Extract gender labels
# Make one-hot sensitive feature columns
ages=ages.values.astype(int)

# Save to pickle files
print("Saving features, labels, and sensitive_attrs to pickle files")
save_pickle(savename_features,x)
save_pickle(savename_gender_labels,gender_labels)
save_pickle(savename_mortal_labels,charlson_labels)
save_pickle(savename_sensitive_attrs,ages)
