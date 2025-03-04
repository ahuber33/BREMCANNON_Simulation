# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:36:52 2025

@author: huber
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_predictions(path, filename, predictions, scale):
    """
    Plot the predictions as a histogram.

    Parameters:
    predictions (np.array): The predictions made by the model.
    """
    # Extraire les valeurs des prédictions
    bin_values_pred = predictions[0]*scale  # Assurez-vous que les prédictions sont sous forme de tableau 1D

    # Créer un tableau d'indices pour les étiquettes des bins
    bin_labels = np.linspace(0, 5000, 501)
    plt.bar(bin_labels[:-1], bin_values_pred, width=np.diff(bin_labels), color='deepskyblue', alpha=0.7)
    plt.xlabel('$\gamma$ energy [keV]')
    plt.ylabel('Predicted N$\gamma_{incident}$ per $\Omega_{detected}$ [/10 keV]')
    plt.ylim(bottom=1)  # Définit la valeur minimale de l'axe Y à 1
    plt.xlim(left=0)  
    plt.yscale('log')
    plt.title(f'Predicted $\gamma$ spectrum from CEDRIC IPs : {filename}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Afficher le graphique
    plt.savefig(path + filename +'.png' , dpi=300, bbox_inches='tight')

    plt.show()
    
