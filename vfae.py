# pytorch_mnist.py 
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.dataset import DataSetLoader
from seldonian.models.pytorch_vae import PytorchVFAE
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    torch.manual_seed(0)
    data_pth = "static/datasets/supervised/adults_vfae/vfae_adults.csv"
    metadata_pth = "static/datasets/supervised/adults_vfae/metadata_vfae.json"
    save_base_dir = 'interface_outputs'
    # save_base_dir='.'
    # Load metadata
    regime='supervised_learning'
    sub_regime='classification'

    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')

    # constraint_strs = ['abs((FNR | [M]) - (FNR | [F])) <= 0.02']
    constraint_strs = ['VAE <= -0.0']
    deltas = [0.05] 
    columns = ["M", "F"]
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime, columns=columns)
    device = torch.device(0)
    model = PytorchVFAE(device, **{"x_dim": 117,
        "s_dim": 1,
        "y_dim": 1,
        "z1_enc_dim": 100,
        "z2_enc_dim": 100,
        "z1_dec_dim": 100,
        "x_dec_dim": 100,
        "z_dim": 50,
        "dropout_rate": 0.0}
    )

    initial_solution_fn = model.get_model_params
    frac_data_in_safety = 0.5

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
            'lambda_init'   : np.array([0.05]),
            'alpha_theta'   : 0.001,
            'alpha_lamb'    : 0.001,
            'beta_velocity' : 0.09,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : True,
            'batch_size'    : 150,
            'n_epochs'      : 10,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        },
    )

    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=False,write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test.")
    else:
        print("Failed safety test")
    st_primary_objective = SA.evaluate_primary_objective(theta=solution,
        branch='safety_test')
    print("Primary objective evaluated on safety test:")
    print(st_primary_objective)