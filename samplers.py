from abc import ABC, abstractmethod
import numpy as np
import random
import gpmp as gp


def f(x, function_name, function_parameters):
    if function_name == "power sinus":
        power = function_parameters[function_name]["Power"]
        return np.sin(2*np.pi*np.sum(x)**power)
    if function_name == "heaviside":
        x_gaps       = function_parameters[function_name]["x gaps"      ]
        y_values     = function_parameters[function_name]["y values"    ]
        sigma_noises = function_parameters[function_name]["sigma noises"]
        for i, (y, sigma_noise) in enumerate(zip(y_values, sigma_noises)):
            if x_gaps[i] <= x[0] <= x_gaps[i+1]:
                return y + np.random.normal(0, sigma_noise)
    if function_name == "multimodal sinus":
        x_gaps       = function_parameters[function_name]["x gaps"      ]
        speed_values = function_parameters[function_name]["speed values"]
        for i, speed_value in enumerate(speed_values):
            if x_gaps[i] <= x[0] <= x_gaps[i+1]:
                return np.sin(2*np.pi*speed_value*np.sum(x))
    if function_name == "Two bumps":
        return gp.misc.testfunctions.twobumps(x)[0]
    if function_name == "Noisy sin":
        sigma_noise = function_parameters[function_name]["sigma noise"]
        return np.sin(2*np.pi*np.sum(x)) + np.random.normal(0, sigma_noise)
    if function_name == "Case 1":
        return np.sin(x) * x
    if function_name == "Case 2":
        return np.sin(x) / x
    
            
def extract_K_distinct_elements_from_samples(K, samples):
    extracted_sample_index = random.sample(range(len(samples)), K)
    extracted_samples      = []
    for k in extracted_sample_index:
        extracted_samples.append(samples[k])
    return extracted_samples


class Sampler(ABC):
    
    @abstractmethod
    def sample(self, N):
        pass
    

class SyntheticSampler(Sampler):
    
    def __init__(self, input_dimension, x_length, function_name, function_parameters):
        self._input_dimension     = input_dimension
        self._x_length            = x_length
        self._function_name       = function_name
        self._function_parameters = function_parameters
        
    def _sort_samples(self, samples):
        if self._input_dimension > 1:
            return samples
        else:
            sorted_samples, xmax_sorted_sample = [], 0
            for _ in range(len(samples)):
                xmin_sample, min_sample = self._x_length, None
                for sample in samples:
                    if sample["x"] <= xmin_sample and sample["x"] > xmax_sorted_sample:
                        xmin_sample, min_sample = sample["x"], sample
                xmax_sorted_sample = xmin_sample
                sorted_samples.append(min_sample)
            return sorted_samples
        
    def _sample_one_point(self):
        x = np.random.rand(self._input_dimension) * self._x_length
        y = f(x, self._function_name, self._function_parameters)
        return {"x" : x, "y" : y}
    
    def sample(self, N):
        return self._sort_samples([self._sample_one_point() for _ in range(N)]) 
    
    def extract_from_samples(self, K, samples):
        return self._sort_samples(extract_K_distinct_elements_from_samples(K, samples))
