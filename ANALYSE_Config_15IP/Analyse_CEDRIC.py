# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:24:30 2025

@author: huber
"""
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import random
import pickle
import joblib
import sys
import uproot  # Pour lire les fichiers ROOT sans PyROOT
import awkward as ak  # uproot utilise awkward pour gérer les structures complexes
import os
# Définir la variable d'environnement pour désactiver oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from scipy.stats.distributions import uniform, skewnorm
import keras
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, PReLU, Dropout, Input
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import ParameterSampler
shuffle_on= True
from sklearn.preprocessing import MinMaxScaler
import Include

############################################################
############### Define for user ############################
############################################################
root_models = 'D:/PYTHON/ANALYSE_Config_15IP/'  #Important to change this PATH for your own use !!!!
root_save = root_models + '/Save/'       #Important to change this PATH for your own use !!!!
keras.__version__

# Charger les scalers
with open(root_models + 'gpy/' + 'scaler_X_Config_GSI2025.pkl', 'rb') as f:
    scaler_X_gpy = pickle.load(f)
with open(root_models + 'nn/' + 'scaler_X_Config_GSI2025.pkl', 'rb') as f:
    scaler_X_nn = pickle.load(f)
with open(root_models + 'rf/' + 'scaler_X_Config_GSI2025.pkl', 'rb') as f:
    scaler_X_rf = pickle.load(f)

with open(root_models + 'gpy/' + 'scaler_Y_Config_GSI2025.pkl', 'rb') as f:
    scaler_Y_gpy = pickle.load(f)
with open(root_models + 'nn/' + 'scaler_Y_Config_GSI2025.pkl', 'rb') as f:
    scaler_Y_nn = pickle.load(f)
with open(root_models + 'rf/' + 'scaler_Y_Config_GSI2025.pkl', 'rb') as f:
    scaler_Y_rf = pickle.load(f)

with open(root_models + 'gpy/' + 'Config_GSI2025.pkl', 'rb') as file:
#with open(root_models + 'gpy/' + 'best.pkl', 'rb') as file:
    nn_model_gauss = pickle.load(file)

with open(root_models + 'rf/' + 'Config_GSI2025.pkl', 'rb') as file:
    rf_model = pickle.load(file)    

nn_model = load_model(root_models + 'nn/' + 'Config_GSI2025.keras')


###########################################################
###########################################################

import tkinter as tk
import numpy as np

# Variables globales
data_array = None
filename = ""  # 🔹 Ajout de filename en global

# Créer la fenêtre principale
root = tk.Tk()
root.title("Data Input")

# Ajouter un champ pour entrer le nom du fichier
tk.Label(root, text="Nom du fichier :").grid(row=0, column=0, padx=10, pady=5, sticky="w")
filename_entry = tk.Entry(root)
filename_entry.grid(row=0, column=1, padx=10, pady=5)

# Paramètres pour le tableau
n_rows = 15
n_cols = 1

# Liste de texte à afficher pour chaque ligne
texts = [f"IP {i+1} [PSL/mm²]" for i in range(n_rows)]

# Liste pour stocker les références aux champs de saisie (entrées)
table = []

# Créer un tableau de champs de saisie avec des descriptions à côté
for i, text in enumerate(texts):
    label = tk.Label(root, text=text)
    label.grid(row=i+1, column=0, padx=10, pady=5, sticky="w")  # Décalage de +1 pour le champ filename

    entry = tk.Entry(root)
    entry.grid(row=i+1, column=1, padx=10, pady=5)

    table.append(entry)  # Stocke chaque champ de saisie

# Fonction pour récupérer les données sous forme de tableau numpy
def get_data():
    global table
    try:
        table_data = np.array([float(entry.get()) for entry in table], dtype=float)
        return table_data
    except ValueError:
        print("Erreur : Assure-toi que toutes les entrées contiennent des nombres valides.")
        return None

# Fonction appelée lors du clic sur "Enregistrer"
def on_save_button_click():
    global data_array, filename  # 🔹 Déclare qu'on utilise la variable globale
    filename = filename_entry.get().strip()  # Récupérer et nettoyer le nom du fichier
    
    if not filename:
        print("Erreur : Veuillez entrer un nom de fichier valide.")
        return

    data_array = get_data()  # Stocke les données dans la variable globale
    
    root.destroy()  # Ferme la fenêtre après enregistrement

# Ajouter un bouton pour enregistrer les résultats
save_button = tk.Button(root, text="Enregistrer", command=on_save_button_click)
save_button.grid(row=n_rows+2, columnspan=n_cols*2, pady=10)

# Lancer l'interface graphique
root.mainloop()



input_norm, integral = Include.normalize_input(data_array)
print(input_norm)
# Appeler la fonction avec le DataFrame
Include.plot_histogram(input_norm)

# Assurez-vous que input_norm est un tableau 1D et le transformer en tableau 2D
# Transformer les inputs
input_data_scaled_nn = scaler_X_nn.transform(input_norm)
input_data_scaled_gpy = scaler_X_gpy.transform(input_norm)
input_data_scaled_rf = scaler_X_rf.transform(input_norm)

# Prédiction GP
predictions_gauss, sigma_predictions_gauss = nn_model_gauss.predict(input_data_scaled_gpy, return_std=True)

# Inverse scaling sur les prédictions GP (espace log10)
predictions_gauss_log = Include.inverse_transform_predictions(scaler_Y_gpy, predictions_gauss)
sigma_predictions_gauss_log = sigma_predictions_gauss * (scaler_Y_gpy.data_range_)

# Retour espace physique
original_predictions_gauss = 10 ** predictions_gauss_log

# Gestion des incertitudes GP
pred_upper_log = predictions_gauss_log + sigma_predictions_gauss_log
pred_lower_log = predictions_gauss_log - sigma_predictions_gauss_log
pred_upper = 10 ** pred_upper_log
pred_lower = 10 ** pred_lower_log
original_sigma_predictions_gauss = pred_upper - original_predictions_gauss

# Prédiction MC Dropout
# Supposons que nn_model est ton modèle entraîné
nn_model_mc = Include.MCDropoutModel(nn_model)
mean_pred, uncertainty_pred, all_simulations = Include.mc_dropout_predictions(
    nn_model_mc, input_data_scaled_nn, n_simulations=100
)

# Appliquer l'inverse scaling dans l'espace log10
simulations_log = np.array([
    Include.inverse_transform_predictions(scaler_Y_nn, sim.reshape(1, -1)).squeeze()
    for sim in all_simulations
])

# Retour à l’espace physique
simulations_phys = 10 ** simulations_log
original_predictions = np.mean(simulations_phys, axis=0)
original_sigma_predictions = np.std(simulations_phys, axis=0)


# Prédiction Random Forest
# Récupération des prédictions de chaque arbre (dans l’espace standardisé log10)
all_simulations_rf = np.array([
    tree.predict(input_data_scaled_rf) for tree in rf_model.estimators_
])  # shape: (n_trees, n_samples)

# Appliquer l'inverse scaling dans l'espace log10
simulations_log_rf = np.array([
    Include.inverse_transform_predictions(scaler_Y_rf, sim.reshape(1, -1)).squeeze()
    for sim in all_simulations_rf
])  # shape: (n_trees, n_samples)

# Retour à l’espace physique (undo log10)
simulations_phys_rf = 10 ** simulations_log_rf  # shape: (n_trees, n_samples)

# Moyenne et écart-type dans l’espace physique
original_predictions_rf = np.mean(simulations_phys_rf, axis=0)
original_sigma_predictions_rf = np.std(simulations_phys_rf, axis=0)

# (Optionnel) calcul des bornes à ±1σ si tu veux les tracer
pred_upper_rf = original_predictions_rf + original_sigma_predictions_rf
pred_lower_rf = original_predictions_rf - original_sigma_predictions_rf


original_sigma_predictions_rf = pred_upper_rf - original_predictions_rf

#print(original_sigma_predictions)
#print(original_sigma_predictions_rf)


Include.plot_predictions(root_save,
                         filename,
                         original_predictions_gauss,
                         original_sigma_predictions_gauss,
                         original_predictions,
                         original_sigma_predictions,
                         original_predictions_rf,
                         original_sigma_predictions_rf,
                         integral)