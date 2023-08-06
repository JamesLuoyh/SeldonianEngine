# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from seldonian.models import objectives
from seldonian.models.pytorch_cnn_vfae import PytorchFacialVAE
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)
import torch

sub_regime = "classification"
N=23700
print("Loading features,labels,sensitive_attrs from file...")

savename_features = '/media/yuhongluo/face_recog/features.pkl'
savename_labels = '/media/yuhongluo/face_recog/labels.pkl'
savename_sensitive_attrs = '/media/yuhongluo/face_recog/sensitive_attrs.pkl'

features = load_pickle(savename_features)
labels = load_pickle(savename_labels)
sensitive_attrs = load_pickle(savename_sensitive_attrs)
print(features.shape)
print(sensitive_attrs.shape)
print(labels.shape)
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
    features=[features,sensitive_attrs[:, :1]],
    labels=labels,
    sensitive_attrs=sensitive_attrs,
    num_datapoints=N,
    meta_information=meta_information)
regime='supervised_learning'
constraint_strs = ['VAE <= 0.01']
deltas = [0.01] 
print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)

device = torch.device(0)
model = PytorchFacialVAE(device, **{"x_dim": -1,
        "s_dim": 1,
        "y_dim": 1,
        "z1_enc_dim": 100,
        "z2_enc_dim": 100,
        "z1_dec_dim": 100,
        "x_dec_dim": 100,
        "z_dim": 100,
        "dropout_rate": 0.0,
        "alpha_adv": 1e-4})

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
        'lambda_init'   : np.array([1.0]),
        'alpha_theta'   : 1e-4,
        'alpha_lamb'    : 1e-4,
        'beta_velocity' : 0.9,
        'beta_rmsprop'  : 0.95,
        'use_batches'   : True,
        'batch_size'    : 237, #237
        'n_epochs'      : 80,
        'gradient_library': "autograd",
        'hyper_search'  : None,
        'verbose'       : True,
    },
    
    batch_size_safety=500
)
save_pickle('/media/yuhongluo/SeldonianExperimentSpecs/facial_recog_spec.pkl',spec,verbose=True)
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

# parse_trees[0].evaluate_constraint(theta=model.get_model_params,dataset=dataset,
# model=model.to("cpu"),regime='supervised_learning',
# branch='safety_test')
# print("VAE constraint", parse_trees[0].root.value)