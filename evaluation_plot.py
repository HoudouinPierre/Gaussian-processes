import numpy as np
import pandas as pd
from scipy.special import erf
import plotly.graph_objects as go
import plotly.express as px


def compute_gaussian_quantiles(mahalanobis_distances):
    gaussian_quantiles = []
    for mahalanobis_distance in mahalanobis_distances:
        gaussian_quantiles.append(erf(mahalanobis_distance / np.sqrt(2)))  
    return gaussian_quantiles
    

def compute_gaussian_theory(samples_type, N=1000):
    if samples_type == "Mahalanobis distance":
        x_gaussian_theory = np.linspace(0, 3, N)
        y_gaussian_theory = [1 - erf(x / np.sqrt(2)) for x in x_gaussian_theory]
    if samples_type == "Gaussian quantile":
        x_gaussian_theory = np.linspace(0, 1, N)
        y_gaussian_theory = [1 - x for x in x_gaussian_theory]
    return x_gaussian_theory, y_gaussian_theory

def empirical_quantile_for_gaussian_quantile(mahalanobis_distances, alpha):
    gaussian_quantiles = compute_gaussian_quantiles(mahalanobis_distances)
    return np.sum(np.array(gaussian_quantiles) < 1 - alpha) / len(gaussian_quantiles)

def plot_empirical_cumulative_distribution_with_plotly(mahalanobis_distances, samples_type):
    x_gaussian_theory, y_gaussian_theory = compute_gaussian_theory(samples_type)
    if samples_type == "Mahalanobis distance":
        samples = mahalanobis_distances
    if samples_type == "Gaussian quantile":
        samples = compute_gaussian_quantiles(mahalanobis_distances)
    df = pd.DataFrame(samples, columns=["Samples"])
    fig = px.ecdf(df, x="Samples", markers=True, marginal="histogram", ecdfmode="reversed")
    fig.add_trace(go.Scatter(x=x_gaussian_theory, y=y_gaussian_theory, mode='lines', name="Gaussian theory"))
    fig.update_xaxes(tickprefix="<b>",ticksuffix="<b><br>")
    fig.update_yaxes(tickprefix="<b>",ticksuffix="<b><br>")
    fig.update_layout(width=1050, height=700, title=samples_type, template="plotly_white")
    fig.show()
    
        
def save_empirical_cumulative_distribution_with_plotly(mahalanobis_distances, samples_type, save_path):
    x_gaussian_theory, y_gaussian_theory = compute_gaussian_theory(samples_type)
    if samples_type == "Mahalanobis distance":
        samples = mahalanobis_distances
    if samples_type == "Gaussian quantile":
        samples = compute_gaussian_quantiles(mahalanobis_distances)
    df = pd.DataFrame(samples, columns=["Samples"])
    fig = px.ecdf(df, x="Samples", markers=True, marginal="histogram", ecdfmode="reversed")
    fig.add_trace(go.Scatter(x=x_gaussian_theory, y=y_gaussian_theory, mode='lines', name="Gaussian theory"))
    fig.update_xaxes(tickprefix="<b>",ticksuffix="<b><br>")
    fig.update_yaxes(tickprefix="<b>",ticksuffix="<b><br>")
    fig.update_layout(width=1500, height=1000, title=samples_type, template="plotly_white")
    fig.write_image(save_path + samples_type + ".png")
