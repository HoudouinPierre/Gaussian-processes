import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as pl
import torch
import gpmp as gp


def plot_samples(X_function, X_observed, Y_function, Y_observed):
    X_function_for_plot, X_observed_for_plot = np.array([x[0] for x in X_function]), np.array([x[0] for x in X_observed])
    fig = go.Figure  ()
    fig.add_trace    (go.Scatter(x=X_function_for_plot, y=Y_function, mode='lines'  , name="f", line=dict(width=1, color="green")))
    fig.add_trace    (go.Scatter(x=X_observed_for_plot, y=Y_observed, mode='markers', name="samples", marker_color="black", marker_size=10, marker_symbol="x"))
    fig.update_xaxes (tickprefix="<b>",ticksuffix="<b><br>",title_text="")
    fig.update_yaxes (tickprefix="<b>",ticksuffix="<b><br>",title_text="")
    fig.update_layout(width=900, height=600, template="plotly_white")
    pyo.iplot        (fig)    
        
    
def plot_prediction(X_function, X_observed, Y_function, Y_observed, posterior_mean, posterior_std):
    X_function_for_plot, X_observed_for_plot = np.array([x[0] for x in X_function]), np.array([x[0] for x in X_observed])
    pl.figure      (figsize=(15,10))
    pl.plot        (X_function_for_plot, Y_function, color="green", label="f"           , linewidth=1)
    pl.scatter     (X_observed_for_plot, Y_observed, color="black", label="Observations", s=100, marker="x")
    pl.fill_between(X_function_for_plot, posterior_mean - 1.96 * posterior_std, posterior_mean + 1.96 * posterior_std, color="lightcoral", alpha=0.3, label="IC à 95%")
    pl.legend(fontsize=15)
    pl.grid()
    axes = pl.gca()
    axes.tick_params(direction='out', length=12, width=3, labelsize=8, grid_alpha=0.5)
    for tickLabel in pl.gca().get_xticklabels() + pl.gca().get_yticklabels():
        tickLabel.set_fontsize(12)


def plot_custom_GPMP(X_function, X_observed, Y_function, Y_observed, posterior_mean, posterior_std, gaussian_process_model):
    X_function_for_plot, X_observed_for_plot = np.array([x[0] for x in X_function]), np.array([x[0] for x in X_observed])
    print('\nVisualization')
    print('-------------')
    fig = gp.misc.plotutils.Figure(isinteractive=True)
    fig.plot                      (X_function_for_plot, Y_function, 'k', linewidth=1, linestyle=(0, (5, 5)))
    fig.plotdata                  (X_observed_for_plot, Y_observed)
    fig.plotgp                    (X_function_for_plot, posterior_mean, posterior_std**2, colorscheme='simple')
    fig.xylabels                  ('$x$', '$z$')
    fig.title                     ('Posterior GP with parameters selected by ReML')
    fig.show                      (grid=True, legend=True, legend_fontsize=9)
    
def plot_custom_GPyTorch(X_function, X_observed, Y_function, Y_observed, posterior_mean, posterior_std):
    X_function_for_plot, X_observed_for_plot = np.array([x[0] for x in X_function]), np.array([x[0] for x in X_observed])
    with torch.no_grad():
        f, ax = pl.subplots(1, 1, figsize=(15, 10))
        ax.plot(X_observed_for_plot, Y_observed, 'k*')
        ax.plot(X_function_for_plot, posterior_mean, 'b')
        ax.fill_between(X_function_for_plot, posterior_mean - 2 * posterior_std, posterior_mean + 2 * posterior_std, alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        
        
def plot_custom(library, X_function, X_observed, Y_function, Y_observed, posterior_mean, posterior_std, gaussian_process_model):
    if library == "GPMP":
        plot_custom_GPMP    (X_function, X_observed, Y_function, Y_observed, posterior_mean, posterior_std, gaussian_process_model)
    if library == "GPyTorch":
        plot_custom_GPyTorch(X_function, X_observed, Y_function, Y_observed, posterior_mean, posterior_std)
        

def plot_evaluation_model(X_function, X_observed_without_sample_i, Y_function, Y_observed_without_sample_i, posterior_mean, posterior_std, X_sample_i, Y_sample_i):
    X_function_for_plot, X_observed_for_plot_without_sample_i = np.array([x[0] for x in X_function]), np.array([x[0] for x in X_observed_without_sample_i])
    pl.figure      (figsize=(15,10))
    pl.plot        (X_function_for_plot                 , Y_function                 , color="green", label="f"           , linewidth=1)
    pl.scatter     (X_observed_for_plot_without_sample_i, Y_observed_without_sample_i, color="black", label="Observations", s=100, marker="x")
    pl.fill_between(X_function_for_plot, posterior_mean - 1.96 * posterior_std, posterior_mean + 1.96 * posterior_std, color="lightcoral", alpha=0.3, label="IC à 95%")
    pl.scatter     (X_sample_i                          , Y_sample_i                 , color="red"  , label="Point i"     , s=200)
    pl.legend(fontsize=15)
    pl.grid()
    axes = pl.gca()
    axes.tick_params(direction='out', length=12, width=3, labelsize=8, grid_alpha=0.5)
    for tickLabel in pl.gca().get_xticklabels() + pl.gca().get_yticklabels():
        tickLabel.set_fontsize(12)

    