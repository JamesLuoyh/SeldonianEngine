# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from facial_recog_cnn import PytorchFacialRecog
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)

import torch

if __name__ == "__main__":
	torch.manual_seed(0)
	regime='supervised_learning'
	sub_regime='classification'
	
	N=23700 # Clips off 5 samples (at random) to make total divisible by 150,
	# the desired batch size
	
	# Get the data, load from file if already saved
	savename_features = './features.pkl'
	savename_labels = './labels.pkl'
	savename_sensitive_attrs = './sensitive_attrs.pkl'
	if not all([os.path.exists(x) for x in [savename_features,
			savename_labels,savename_sensitive_attrs]]):
		print("loading data...")
		data = pd.read_csv('../../../facial_recognition/Kaggle_UTKFace/age_gender.csv')
		# Shuffle data since it is in order of age, then gender
		data = data.sample(n=len(data),random_state=42).iloc[:N]
		print("Randomly sampled data, first 20 points")
		print(data.iloc[0:20])
		# Convert pixels from string to numpy array
		print("Converting pixels to array...")
		data['pixels']=data['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))

		# normalize pixels data
		print("Normalizing and reshaping pixel data...")
		data['pixels'] = data['pixels'].apply(lambda x: x/255)

		# Reshape pixels array
		X = np.array(data['pixels'].tolist())
		# Converting pixels from 1D to 4D
		features = X.reshape(X.shape[0],1,48,48)
		labels = data['gender'].values
		M=data['gender'].values
		mask=~(M.astype("bool"))
		F=mask.astype('int64')
		sensitive_attrs = np.hstack((M.reshape(-1,1),F.reshape(-1,1)))
		print("Saving features, labels, and sensitive_attrs to pickle files")
		save_pickle(savename_features,features)
		save_pickle(savename_labels,labels)
		save_pickle(savename_sensitive_attrs,sensitive_attrs)
	else:
		print("Loading features,labels,sensitive_attrs from file...")
		features = load_pickle(savename_features)
		labels = load_pickle(savename_labels)
		sensitive_attrs = load_pickle(savename_sensitive_attrs)
	
	assert len(features) == N
	assert len(labels) == N
	assert len(sensitive_attrs) == N
	frac_data_in_safety = 0.5
	sensitive_col_names = ['M','F']

	meta_information = {}
	meta_information['feature_col_names'] = ['img']
	meta_information['label_col_names'] = ['label']
	meta_information['sensitive_col_names'] = sensitive_col_names
	meta_information['sub_regime'] = sub_regime
	
	print("Making SupervisedDataSet...")
	dataset = SupervisedDataSet(
		features=features,
		labels=labels,
		sensitive_attrs=sensitive_attrs,
		num_datapoints=N,
		meta_information=meta_information)

	constraint_strs = ['min((ACC | [M])/(ACC | [F]),(ACC | [F])/(ACC | [M])) >= 0.8']
	deltas = [0.05] 
	print("Making parse trees for constraint(s):")
	print(constraint_strs," with deltas: ", deltas)
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,deltas,regime=regime,
		sub_regime=sub_regime,columns=sensitive_col_names)
	device = torch.device("mps")
	model = PytorchFacialRecog(device)


	initial_solution_fn = model.get_model_params
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=objectives.binary_logistic_loss,
		use_builtin_primary_gradient_fn=False,
		sub_regime=sub_regime,
		initial_solution_fn=initial_solution_fn,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.001,
			'alpha_lamb'    : 0.001,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'use_batches'   : True,
			'batch_size'    : 237,
			'n_epochs'      : 40,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
		batch_size_safety=2000
	)
	save_pickle('./spec.pkl',spec,verbose=True)
	SA = SeldonianAlgorithm(spec)


	passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
	if passed_safety:
		print("Passed safety test")
		st_primary_objective = SA.evaluate_primary_objective(theta=solution,
		branch='safety_test')
		print("Primary objective evaluated on safety test:")
		print(st_primary_objective)
	else:
		print("Failed safety test")