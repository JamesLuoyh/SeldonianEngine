# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from seldonian.models import objectives
from seldonian.models.pytorch_vae import PytorchVFAE
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)
import torch

sub_regime = "classification"
# N=23700
print("Loading features,labels,sensitive_attrs from file...")

savename_features = '/media/yuhongluo/health/features.pkl'
savename_labels = '/media/yuhongluo/health/gender_labels.pkl'
savename_sensitive_attrs = '/media/yuhongluo/health/sensitive_attrs.pkl'

features = load_pickle(savename_features)
labels = load_pickle(savename_labels)
sensitive_attrs = load_pickle(savename_sensitive_attrs)
print(features.shape)
print(sensitive_attrs.shape)
print(labels.shape)

frac_data_in_safety = 0.5
sensitive_col_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

meta_information = {}
meta_information['feature_col_names'] = ['img']
meta_information['label_col_names'] = ['label']
meta_information['sensitive_col_names'] = sensitive_col_names
meta_information['sub_regime'] = sub_regime
print("Making SupervisedDataSet...")
dataset = SupervisedDataSet(
    features=np.concatenate((features,sensitive_attrs,np.expand_dims(labels, axis=1)), -1),
    labels=labels,
    sensitive_attrs=sensitive_attrs,
    num_datapoints=features.shape[0],
    meta_information=meta_information)
regime='supervised_learning'
batch_size_safety=100
constraint_strs = ['VAE <= 0.05']
deltas = [0.02] 
print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)

device = torch.device(0)
model = PytorchVFAE(device, **{"x_dim": features.shape[1],
        "s_dim": sensitive_attrs.shape[1],
        "y_dim": 1,
        "z1_enc_dim": 100,
        "z2_enc_dim": 100,
        "z1_dec_dim": 100,
        "x_dec_dim": 100,
        "z_dim": 100,
        "dropout_rate": 0.0,
        "alpha_adv": 1e-3})

initial_solution_fn = model.get_model_params
spec = SupervisedSpec(
    dataset=dataset,
    model=model,
    parse_trees=parse_trees,
    frac_data_in_safety=frac_data_in_safety,
    primary_objective=objectives.vae_loss,
    use_builtin_primary_gradient_fn=False,
    sub_regime=sub_regime,
    initial_solution_fn=initial_solution_fn,
    optimization_technique='gradient_descent',
    optimizer='adam',
    optimization_hyperparams={
        'lambda_init'   : np.array([0.5]),
        'alpha_theta'   : 1e-4,
        'alpha_lamb'    : 1e-4,
        'beta_velocity' : 0.9,
        'beta_rmsprop'  : 0.95,
        'use_batches'   : True,
        'batch_size'    : 200, #237
        'n_epochs'      : 30,
        'gradient_library': "autograd",
        'hyper_search'  : None,
        'verbose'       : True,
        'n_adv_rounds'  : 1,
    },
    
    batch_size_safety=200
)
save_pickle('/media/yuhongluo/SeldonianExperimentSpecs/health_spec.pkl',spec,verbose=True)
SA = SeldonianAlgorithm(spec)
passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
if passed_safety:
    print("Passed safety test.")
else:
    print("Failed safety test")
st_primary_objective = SA.evaluate_primary_objective(theta=solution,
    branch='safety_test')
print("Primary objective evaluated on safety test:")
print(st_primary_objective)

parse_trees[0].evaluate_constraint(theta=model.get_model_params,dataset=dataset,
model=model,regime='supervised_learning',
branch='safety_test',
batch_size_safety=batch_size_safety)
print("VAE constraint", parse_trees[0].root.value)