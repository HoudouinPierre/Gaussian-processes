import numpy as np
import torch


def preprocessingGPMP(function_samples, observed_samples):
    X_function = np.array([sample["x"] for sample in function_samples])
    X_observed = np.array([sample["x"] for sample in observed_samples]) 
    Y_function = np.array([sample["y"] for sample in function_samples])
    Y_observed = np.array([sample["y"] for sample in observed_samples])
    return X_function, X_observed, Y_function, Y_observed


def preprocessingSKlearn(function_samples, observed_samples):
    X_function = np.array([sample["x"] for sample in function_samples])
    X_observed = np.array([sample["x"] for sample in observed_samples])
    Y_function = np.array([sample["y"] for sample in function_samples])
    Y_observed = np.array([sample["y"] for sample in observed_samples])
    return X_function, X_observed, Y_function, Y_observed


def preprocessingGPyTorch(function_samples, observed_samples):
    X_function = torch.from_numpy(np.array([sample["x"] for sample in function_samples]))
    X_observed = torch.from_numpy(np.array([sample["x"] for sample in observed_samples]))
    Y_function = torch.from_numpy(np.array([sample["y"] for sample in function_samples]))
    Y_observed = torch.from_numpy(np.array([sample["y"] for sample in observed_samples]))
    return X_function, X_observed, Y_function, Y_observed


def preprocessing(library, function_samples, observed_samples):
    if library == "GPMP":
        return preprocessingGPMP   (function_samples, observed_samples)
    if library == "SKlearn":
        return preprocessingSKlearn(function_samples, observed_samples)
    if library == "GPyTorch":
        return preprocessingGPyTorch(function_samples, observed_samples)
