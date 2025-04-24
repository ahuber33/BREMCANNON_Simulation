# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:36:52 2025

@author: huber
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model

class MCDropoutModel(Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=True)  # Dropout reste actif même en mode "inference"


def mc_dropout_predictions(model, input_data, n_simulations=100):
    simulations = []
    for _ in range(n_simulations):
        predictions = model(input_data, training=True)  # Force dropout
        simulations.append(predictions.numpy().squeeze())

    simulations = np.array(simulations)
    mean_prediction = np.mean(simulations, axis=0)
    std_prediction = np.std(simulations, axis=0)
    
    return mean_prediction, std_prediction, simulations

# Fonction pour enregistrer les résultats dans un fichier
def save_results(data):
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, "output_results.txt")
    np.savetxt(filename, data, delimiter='\t', fmt="%.6f")
    print(f"Fichier sauvegardé : {filename}")

def normalize_input(results_df):
    # Convertir en tableau numpy et aplatir en 1D
    bin_values = np.array(results_df, dtype=float).flatten()
    
    # Calculer l'intégrale (la somme des bins)
    integral_value = np.sum(bin_values)

    # Normaliser les données en divisant par l'intégrale
    normalized_data = bin_values / integral_value

    # Créer les noms des colonnes, par exemple "PSL_IP_1", "PSL_IP_2", etc.
    columns = [f"PSL_IP_{i+1}" for i in range(len(normalized_data))]

    # Créer un DataFrame avec les données normalisées
    df_normalized = pd.DataFrame([normalized_data], columns=columns)
    
    return df_normalized, integral_value

def plot_histogram(results_df):
    """
    Plot a histogram from the normalized bin values in a DataFrame with error bars.

    Parameters:
    results_df (pd.DataFrame): A DataFrame containing the normalized bin values.
    uncert_df (pd.DataFrame): A DataFrame containing the uncertainties for each bin.
    """
    # Extraire les valeurs et les étiquettes du DataFrame
    # Assurer que bin_values contient des nombres
    bin_values = np.array(results_df, dtype=float).flatten()  # Convertir en float et aplatir en 1D
    bin_labels = [f"IP {i+1}" for i in range(len(bin_values))]  # Générer les labels "IP 1", "IP 2", ...


    # Tracer l'histogramme avec des barres d'erreur
    plt.figure(figsize=(10, 6))
    plt.bar(bin_labels, bin_values, color='skyblue', alpha=0.7, capsize=5)
    plt.xlabel('Bin Labels')
    plt.ylabel('Normalized Bin Values')
    plt.title('Histogram of Normalized Bin Values with Uncertainties')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    


def inverse_transform_predictions(scaler_y, predictions):
    """
    Inverse transform the predictions using the provided MinMaxScaler for y.

    Parameters:
    scaler_y (MinMaxScaler): The MinMaxScaler object used for normalizing y.
    predictions (np.array): The predictions made by the model.

    Returns:
    np.array: The inverse transformed predictions.
    """
    # Inverser la transformation pour les prédictions
    original_predictions = scaler_y.inverse_transform(predictions)
    return original_predictions

def plot_predictions(path, filename, original_predictions_gauss, original_sigma_predictions_gauss, predictions, sigma_predictions, predictions_rf, sigma_predictions_rf, scale):
    """
    Plot the predictions as a histogram.

    Parameters:
    predictions (np.array): The predictions made by the model.
    """
    bin_values_pred = predictions * scale
    bin_err_pred = sigma_predictions * scale
    bin_values_pred_gauss = original_predictions_gauss[0] * scale
    bin_err_pred_gauss = original_sigma_predictions_gauss[0] * scale
    bin_values_pred_rf = predictions_rf * scale
    bin_err_pred_rf = sigma_predictions_rf * scale
    bin_labels = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 
              150, 250, 350, 450, 550, 650, 750, 850, 950, 
              1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500,
              12500, 17500, 22500, 27500, 32500, 37500, 42500, 47500, 52500, 57500, 62500, 67500, 72500, 77500, 82500, 87500, 92500, 97500,
              125000, 175000, 225000, 275000, 325000, 375000, 425000, 475000]
    bin_errors = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
              50, 50, 50, 50, 50, 50, 50, 50, 50, 
              500, 500, 500, 500, 500, 500, 500, 500, 500,
              2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,
              25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000]

    # Créer un tableau d'indices pour les étiquettes des bins
    plt.errorbar(bin_labels, bin_values_pred, xerr=bin_errors, yerr=bin_err_pred, fmt='o', capsize=5,
                color='blue', alpha=0.8, linewidth=0.8, 
                label = rf"Sequential-gelu model")


    plt.errorbar(bin_labels, bin_values_pred_gauss, xerr=bin_errors, yerr=bin_err_pred_gauss, fmt='o', capsize=5,
                    color='red', alpha=0.8, linewidth=0.8, 
                    label = rf"Gaussian Process model :")
    
    plt.errorbar(bin_labels, bin_values_pred_rf, xerr=bin_errors, yerr=bin_err_pred_rf, fmt='o', capsize=5,
                    color='green', alpha=0.8, linewidth=0.8, 
                    label = rf"Random Forest model :")
    
    plt.fill_between(bin_labels, bin_values_pred_gauss - bin_err_pred_gauss, bin_values_pred_gauss + bin_err_pred_gauss, color='red', alpha=0.4)
    plt.fill_between(bin_labels, bin_values_pred - bin_err_pred, bin_values_pred + bin_err_pred, color='blue', alpha=0.25)
    plt.fill_between(bin_labels, bin_values_pred_rf - bin_err_pred_rf, bin_values_pred_rf + bin_err_pred_rf, color='green', alpha=0.15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.xlabel(r'$\gamma$ energy [keV]')
    plt.ylabel(r'Predicted N$\gamma_{incident}$ per $\Omega_{detected}$ [/10 keV]')
    plt.ylim(bottom=1)  # Définit la valeur minimale de l'axe Y à 1
    plt.xlim(left=1)  
    plt.yscale('log')
    plt.xscale('log')
    plt.title(rf'Predicted $\gamma$ spectrum from CEDRIC IPs : {filename}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    
    # Afficher le graphique
    plt.savefig(path + filename +'.png' , dpi=300, bbox_inches='tight')

    plt.show()
    
