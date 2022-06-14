import os,sys
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.seldonian_algorithm import seldonian_algorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
	# gpa dataset
	interface_output_dir = os.path.join('/Users/ahoag/beri/code',
		'interface_outputs/demographic_parity')
	specfile = os.path.join(interface_output_dir,'spec.pkl')
	spec = load_pickle(specfile)
	spec.primary_objective = spec.model_class().sample_logistic_loss
	passed_safety,candidate_solution = seldonian_algorithm(spec)
	print(passed_safety,candidate_solution)
