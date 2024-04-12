import torch
import numpy as np
from gaussian_process_model import GaussianProcessModel
from gaussian_process_plot import plot_evaluation_model


class GaussianProcessEvaluation:
    
    def __init__(self, library, X_function, Y_function, X_observed, Y_observed, kernel_name, kernel_parameters_optimization, gaussian_process_parameters, show_plots=False):
        self._library                        = library
        self._X_function                     = X_function
        self._Y_function                     = Y_function
        self._X_observed                     = X_observed
        self._Y_observed                     = Y_observed
        self._kernel_name                    = kernel_name
        self._kernel_parameters_optimization = kernel_parameters_optimization
        self._gaussian_process_parameters    = gaussian_process_parameters
        self._show_plots                     = show_plots
        
    def _fill_learnt_gaussian_process_parameters(self):
        self._kernel_parameters_optimization["sigma0"] = self._gaussian_process_parameters["sigma0"]
        self._kernel_parameters_optimization["Noise" ] = self._gaussian_process_parameters["Noise" ]
        if self._gaussian_process_parameters.get("Length scale"):
            self._kernel_parameters_optimization["Length scale"] = self._gaussian_process_parameters["Length scale"]
        if self._gaussian_process_parameters.get("Length scale 0"):
            length_scales = []
            for length_scale_index in range(len(self._X_observed[0])):
                length_scales.append(self._gaussian_process_parameters["Length scale " + str(length_scale_index)])
            self._kernel_parameters_optimization["Length scales"] = length_scales
            
    def _split_X_observed_and_Y_observed_by_removing_sample_i(self, i):
        if self._library == "GPMP":
            X_observed_without_sample_i = np.concatenate((self._X_observed[:i], self._X_observed[i+1:]))
            Y_observed_without_sample_i = np.concatenate((self._Y_observed[:i], self._Y_observed[i+1:]))
            X_sample_i                  = np.array([list(self._X_observed[i])])
            Y_sample_i                  = self._Y_observed[i]
        if self._library == "SKlearn":
            X_observed_without_sample_i = np.concatenate((self._X_observed[:i], self._X_observed[i+1:]))
            Y_observed_without_sample_i = np.concatenate((self._Y_observed[:i], self._Y_observed[i+1:]))
            X_sample_i                  = np.array([list(self._X_observed[i])])
            Y_sample_i                  = self._Y_observed[i]
        if self._library == "GPyTorch":
            X_observed_without_sample_i = torch.concat((self._X_observed[:i], self._X_observed[i+1:]))
            Y_observed_without_sample_i = torch.concat((self._Y_observed[:i], self._Y_observed[i+1:]))
            X_sample_i                  = torch.tensor(np.array([list(self._X_observed[i].detach().numpy())]))
            Y_sample_i                  = self._Y_observed[i].item()
        return X_observed_without_sample_i, Y_observed_without_sample_i, X_sample_i, Y_sample_i
        
    def _compute_mahalanobis_distance_for_sample_i(self, i):
        X_observed_without_sample_i, Y_observed_without_sample_i, X_sample_i, Y_sample_i = self._split_X_observed_and_Y_observed_by_removing_sample_i(i)
        optimize_kernel_hyperparameters                 = False
        gaussian_process_model_without_sample_i         = GaussianProcessModel(self._library, X_observed_without_sample_i, Y_observed_without_sample_i, self._kernel_name, self._kernel_parameters_optimization, optimize_kernel_hyperparameters)
        gaussian_process_model_without_sample_i.train_gaussian_process_model()
        posterior_mean_sample_i, posterior_std_sample_i = gaussian_process_model_without_sample_i.predict(X_sample_i)
        posterior_std_sample_i                          = [max(posterior_std_sample_i[0], 1e-4)]
        mahalanobis_distance                            = np.abs(posterior_mean_sample_i[0] - Y_sample_i) / posterior_std_sample_i[0]
        posterior_mean, posterior_std                   = gaussian_process_model_without_sample_i.predict(self._X_function)
        if self._show_plots:
            plot_evaluation_model(self._X_function, X_observed_without_sample_i, self._Y_function, Y_observed_without_sample_i, posterior_mean, posterior_std, X_sample_i, Y_sample_i)
        return mahalanobis_distance
    
    def compute_mahalanobis_distances(self):
        self._fill_learnt_gaussian_process_parameters()
        mahalanobis_distances = []
        for i in range(len(self._X_observed)):
            mahalanobis_distances.append(self._compute_mahalanobis_distance_for_sample_i(i))
        return mahalanobis_distances
