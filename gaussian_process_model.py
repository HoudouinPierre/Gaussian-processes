import numpy as np
import gpmp.num as gnp
import gpmp as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
import torch
import gpytorch

# C:\Users\Public\anaconda_file\envs\py38\lib\site-packages\gpmp\
    
class GaussianProcessGPMP:
    
    def __init__(self, X_observed, Y_observed, kernel_name, kernel_parameters_optimization, optimize_kernel_hyperparameters):
        self._X_observed                      = X_observed
        self._Y_observed                      = Y_observed
        self._kernel_name                     = kernel_name
        self._kernel_parameters_optimization  = kernel_parameters_optimization
        self._optimize_kernel_hyperparameters = optimize_kernel_hyperparameters
        self._mean                            = None
        self._kernel                          = None
        self._covparam0                       = None
        self._gaussian_process_model          = None
        self.gaussian_process_parameters      = {}
            
    def _create_mean(self):
        def constant_mean(x, param):
            return gnp.ones((x.shape[0], 1))
        self._mean = constant_mean
        
    def _create_kernel(self):
        def kernel(x, y, covparam, pairwise=False):
            noise = self._kernel_parameters_optimization["Noise"]
            if self._kernel_name == "RBF":
                p = 50
            if self._kernel_name == "Exponential":
                p = 0
            if self._kernel_name == "Matern 3/2":
                p = 1
            if self._kernel_name == "Matern 5/2":
                p = 2
            return gp.kernel.maternp_covariance(x, y, p, covparam, noise=noise, pairwise=pairwise)
        self._kernel = kernel

    def _initialize_gaussian_process_model(self):
        sigma0       = self._kernel_parameters_optimization["sigma0"      ]
        if len(self._kernel_parameters_optimization["Length scales"]) == len(self._X_observed[0]):
            self._covparam0     = torch.tensor([2*np.log(sigma0)])
            for length_scale in self._kernel_parameters_optimization["Length scales"]:
                self._covparam0 = torch.concat((self._covparam0, torch.tensor([-np.log(length_scale)])))
        else:
            length_scale       = self._kernel_parameters_optimization["Length scale"]
            self._covparam0    = torch.tensor([2*np.log(sigma0), -np.log(length_scale)])
        self._gaussian_process_model = gp.core.Model(self._mean, self._kernel, covparam=self._covparam0)
        
    def _extract_gaussian_process_model_parameters(self):
        self.gaussian_process_parameters["sigma0"      ] = np.exp(0.5*self._gaussian_process_model.covparam[0].item())
        self.gaussian_process_parameters["Noise"       ] = self._kernel_parameters_optimization["Noise"]
        for length_scale_index, length_scale in enumerate(self._gaussian_process_model.covparam[1:]):    
            self.gaussian_process_parameters["Length scale " + str(length_scale_index)] = np.exp(   -length_scale.item())
        
    def train_gaussian_process_model(self):
        self._create_mean()
        self._create_kernel()
        self._initialize_gaussian_process_model()
        if self._optimize_kernel_hyperparameters:
            self._gaussian_process_model = gp.kernel.select_parameters_with_reml(self._gaussian_process_model, self._X_observed, self._Y_observed, covparam0=self._covparam0, info=False)
        self._extract_gaussian_process_model_parameters()
        
    def predict(self, X_function):
        posterior_mean, posterior_std = self._gaussian_process_model.predict(self._X_observed, self._Y_observed, X_function)
        return posterior_mean, posterior_std  


class GaussianProcessSKlearn:
    
    def __init__(self, X_observed, Y_observed, kernel_name, kernel_parameters_optimization, optimize_kernel_hyperparameters):
        self._X_observed                      = X_observed
        self._Y_observed                      = Y_observed
        self._kernel_name                     = kernel_name
        self._kernel_parameters_optimization  = kernel_parameters_optimization
        self._optimize_kernel_hyperparameters = optimize_kernel_hyperparameters
        self._kernel                          = None
        self._gaussian_process_model          = None
        self.gaussian_process_parameters      = {}
        
    def _create_kernel(self):
        sigma0                   = self._kernel_parameters_optimization["sigma0"                  ]
        length_scale             = self._kernel_parameters_optimization["Length scale"            ]
        noise                    = self._kernel_parameters_optimization["Noise"                   ]
        lower_bound_sigma0       = self._kernel_parameters_optimization["Lower bound sigma0"      ]
        upper_bound_sigma0       = self._kernel_parameters_optimization["Upper bound sigma0"      ]
        lower_bound_length_scale = self._kernel_parameters_optimization["Lower bound length scale"]
        upper_bound_length_scale = self._kernel_parameters_optimization["Upper bound length scale"]
        lower_bound_noise        = self._kernel_parameters_optimization["Lower bound noise"       ]
        upper_bound_noise        = self._kernel_parameters_optimization["Upper bound noise"       ]
        self._kernel             = 1 * RBF(length_scale=1.0)
        self._kernel.k1          = ConstantKernel(constant_value=sigma0**2, constant_value_bounds=(lower_bound_sigma0**2, upper_bound_sigma0**2)) 
        if self._kernel_name == "RBF":
            self._kernel.k2     = RBF(length_scale=length_scale, length_scale_bounds=(lower_bound_length_scale, upper_bound_length_scale))
        if self._kernel_name == "Exponential":
            self._kernel.k2     = Matern(nu=0.5, length_scale=length_scale, length_scale_bounds=(lower_bound_length_scale, upper_bound_length_scale))
        if self._kernel_name == "Matern 3/2":
            self._kernel.k2     = Matern(nu=1.5, length_scale=length_scale, length_scale_bounds=(lower_bound_length_scale, upper_bound_length_scale))
        if self._kernel_name == "Matern 5/2":
            self._kernel.k2     = Matern(nu=2.5, length_scale=length_scale, length_scale_bounds=(lower_bound_length_scale, upper_bound_length_scale))
        self._kernel             = self._kernel +  WhiteKernel(noise_level=noise, noise_level_bounds=(lower_bound_noise, upper_bound_noise))
        
    def _initialize_gaussian_process_model(self):
        n_restarts_optimizer         = self._kernel_parameters_optimization["SKlearn"]["N restarts optimizer"]
        self._gaussian_process_model = GaussianProcessRegressor(kernel=self._kernel, n_restarts_optimizer=n_restarts_optimizer)
    
    def _update_gaussian_process_model_kernel(self):
        sigma0                                         = self._kernel_parameters_optimization["sigma0"      ]
        length_scale                                   = self._kernel_parameters_optimization["Length scale"]
        noise                                          = self._kernel_parameters_optimization["Noise"       ]
        self._gaussian_process_model.kernel_.k1.k1     = ConstantKernel(constant_value=sigma0**2) 
        if self._kernel_name == "RBF":
            self._gaussian_process_model.kernel_.k1.k2 = RBF(length_scale=length_scale)
        if self._kernel_name == "Exponential":
            self._gaussian_process_model.kernel_.k1.k2 = Matern(nu=0.5, length_scale=length_scale)
        if self._kernel_name == "Matern 3/2":
            self._gaussian_process_model.kernel_.k1.k2 = Matern(nu=1.5, length_scale=length_scale)
        if self._kernel_name == "Matern 5/2":
            self._gaussian_process_model.kernel_.k1.k2 = Matern(nu=2.5, length_scale=length_scale)
        self._gaussian_process_model.kernel_.k2        = WhiteKernel(noise_level=noise)
        
    def _extract_gaussian_process_model_parameters(self):
        self.gaussian_process_parameters["sigma0"      ] = np.sqrt(self._gaussian_process_model.kernel_.get_params()["k1__k1__constant_value"])
        self.gaussian_process_parameters["Noise"       ] =         self._gaussian_process_model.kernel_.get_params()["k2__noise_level"       ]
        self.gaussian_process_parameters["Length scale"] =         self._gaussian_process_model.kernel_.get_params()["k1__k2__length_scale"  ]
        
    def train_gaussian_process_model(self):
        self._create_kernel()
        self._initialize_gaussian_process_model()
        self._gaussian_process_model.fit(self._X_observed, self._Y_observed)
        if not self._optimize_kernel_hyperparameters:
            self._update_gaussian_process_model_kernel()
        self._extract_gaussian_process_model_parameters()

    def predict(self, X_function):
        posterior_mean, posterior_std = self._gaussian_process_model.predict(X_function, return_std=True)
        return posterior_mean, posterior_std           
    

class GaussianProcessGPyTorch:
    
    def __init__(self, X_observed, Y_observed, kernel_name, kernel_parameters_optimization, optimize_kernel_hyperparameters):
        self._X_observed                      = X_observed
        self._Y_observed                      = Y_observed
        self._kernel_name                     = kernel_name
        self._kernel_parameters_optimization  = kernel_parameters_optimization
        self._optimize_kernel_hyperparameters = optimize_kernel_hyperparameters
        self._mean                            = None
        self._kernel                          = None
        self._likelihood                      = None
        self._gaussian_process_model          = None
        self.gaussian_process_parameters      = {}
        
    def _create_mean(self):
        self._mean = gpytorch.means.ConstantMean()
    
    def _create_kernel(self):
        sigma0                   = self._kernel_parameters_optimization["sigma0"                  ]
        if len(self._kernel_parameters_optimization["Length scales"]) == len(self._X_observed[0]):
            length_scale         = self._kernel_parameters_optimization["Length scales"           ]
        else:
            length_scale         = self._kernel_parameters_optimization["Length scale"            ]
        lower_bound_sigma0       = self._kernel_parameters_optimization["Lower bound sigma0"      ]
        upper_bound_sigma0       = self._kernel_parameters_optimization["Upper bound sigma0"      ]
        lower_bound_length_scale = self._kernel_parameters_optimization["Lower bound length scale"]
        upper_bound_length_scale = self._kernel_parameters_optimization["Upper bound length scale"]
        dimension                = len(self._X_observed[0])
        if self._kernel_name == "RBF":
            self._kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dimension, lengthscale_constraint=gpytorch.constraints.Interval(lower_bound_length_scale, upper_bound_length_scale)), outputscale_constraint=gpytorch.constraints.Interval(lower_bound_sigma0, upper_bound_sigma0))
        if self._kernel_name == "Exponential":
            self._kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=dimension, lengthscale_constraint=gpytorch.constraints.Interval(lower_bound_length_scale, upper_bound_length_scale)), outputscale_constraint=gpytorch.constraints.Interval(lower_bound_sigma0, upper_bound_sigma0))
        if self._kernel_name == "Matern 3/2":
            self._kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=dimension, lengthscale_constraint=gpytorch.constraints.Interval(lower_bound_length_scale, upper_bound_length_scale)), outputscale_constraint=gpytorch.constraints.Interval(lower_bound_sigma0, upper_bound_sigma0))
        if self._kernel_name == "Matern 5/2":
            self._kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=dimension, lengthscale_constraint=gpytorch.constraints.Interval(lower_bound_length_scale, upper_bound_length_scale)), outputscale_constraint=gpytorch.constraints.Interval(lower_bound_sigma0, upper_bound_sigma0))
        self._kernel.outputscale             = torch.tensor(sigma0)
        self._kernel.base_kernel.lengthscale = torch.tensor(length_scale)  

    def _create_likelihood(self):
        noise             = self._kernel_parameters_optimization["Noise"            ]
        lower_bound_noise = self._kernel_parameters_optimization["Lower bound noise"]
        upper_bound_noise = self._kernel_parameters_optimization["Upper bound noise"]
        self._likelihood  = gpytorch.likelihoods.GaussianLikelihood()
        self._likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(lower_bound_noise, upper_bound_noise))
        self._likelihood.noise_covar.noise = torch.tensor(noise)
        
    def _initialize_gaussian_process_model(self):
        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module  = mean_module
                self.covar_module = covar_module
            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))
        self._gaussian_process_model = GPModel(self._X_observed, self._Y_observed, self._likelihood, self._mean, self._kernel)
        
    def _train(self):
        self._gaussian_process_model.train()
        self._likelihood            .train()
        training_iter = self._kernel_parameters_optimization["GPyTorch"]["Training iterations"]
        optimizer     = torch.optim.Adam(self._gaussian_process_model.parameters(), lr=0.1)
        mll           = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._gaussian_process_model)
        for i in range(training_iter):
            optimizer.zero_grad()
            loss      = -mll(self._gaussian_process_model(self._X_observed), self._Y_observed)
            loss     .backward()
            optimizer.step()
        
    def _extract_gaussian_process_model_parameters(self):
        self.gaussian_process_parameters["sigma0"] = self._kernel.outputscale.item()
        self.gaussian_process_parameters["Noise" ] = self._likelihood.noise_covar.noise.item()
        for length_scale_index, length_scale in enumerate( self._kernel.base_kernel.lengthscale[0]):    
            self.gaussian_process_parameters["Length scale " + str(length_scale_index)] = length_scale.item()
            
    def train_gaussian_process_model(self):
        self._create_mean()
        self._create_kernel()
        self._create_likelihood()
        self._initialize_gaussian_process_model()
        if self._optimize_kernel_hyperparameters:
            self._train()
        self._extract_gaussian_process_model_parameters()
        
    def predict(self, X_function):
        self._gaussian_process_model.eval()
        self._likelihood            .eval()
        y_preds                       = self._likelihood(self._gaussian_process_model(X_function))
        posterior_mean, posterior_std = y_preds.mean, torch.sqrt(y_preds.variance)
        return posterior_mean.detach().numpy(), posterior_std.detach().numpy()


class GaussianProcessModel:
    
    def __init__(self, library, X_observed, Y_observed, kernel_name, kernel_parameters_optimization, optimize_kernel_hyperparameters):
        self._library                         = library
        self._X_observed                      = X_observed
        self._Y_observed                      = Y_observed
        self._kernel_name                     = kernel_name
        self._kernel_parameters_optimization  = kernel_parameters_optimization
        self._optimize_kernel_hyperparameters = optimize_kernel_hyperparameters
        self._gaussian_process_model          = None
        self.gaussian_process_parameters      = None
        
    def train_gaussian_process_model(self):
        if self._library == "GPMP":
            self._gaussian_process_model = GaussianProcessGPMP   (self._X_observed, self._Y_observed, self._kernel_name, self._kernel_parameters_optimization, self._optimize_kernel_hyperparameters)
        if self._library == "SKlearn":
            self._gaussian_process_model = GaussianProcessSKlearn(self._X_observed, self._Y_observed, self._kernel_name, self._kernel_parameters_optimization, self._optimize_kernel_hyperparameters)
        if self._library == "GPyTorch":
            self._gaussian_process_model = GaussianProcessGPyTorch(self._X_observed, self._Y_observed, self._kernel_name, self._kernel_parameters_optimization, self._optimize_kernel_hyperparameters)
        self._gaussian_process_model.train_gaussian_process_model()
        self.gaussian_process_parameters = self._gaussian_process_model.gaussian_process_parameters
            
    def predict(self, X_function):
        posterior_mean, posterior_std = self._gaussian_process_model.predict(X_function)
        return posterior_mean, posterior_std
