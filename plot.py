from seldonian.utils.io_utils import load_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load loan spec file
    cs_file = '/media/yuhongluo/SeldonianEngine/logs/candidate_selection_log107.p'
    solution_dict = load_pickle(cs_file)
    
    fig = plot_gradient_descent(solution_dict,
        primary_objective_name='vae loss',
        save=False)
    plt.savefig('/media/yuhongluo/SeldonianExperimentResults/cnn_0.02.png')