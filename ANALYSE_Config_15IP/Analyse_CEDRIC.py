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
root_models = '/mnt/d/Simulations/BREMCANNON_Simulation/ANALYSE_Config_15IP/'  #Important to change this PATH for your own use !!!!
root_save = root_models + 'Save/'
keras.__version__

# Charger les scalers
with open(root_models + 'gpy/' + 'scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

with open(root_models + 'gpy/' + 'scaler_Y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)
    
with open(root_models + 'gpy/' + 'best.pkl', 'rb') as f:
    nn_model = pickle.load(f)    


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
save_button = tk.Button(root, text="Proceed", command=on_save_button_click)
save_button.grid(row=n_rows+2, columnspan=n_cols*2, pady=10)

# Lancer l'interface graphique
root.mainloop()



input_norm, integral = Include.normalize_input(data_array)
print(input_norm)
# Appeler la fonction avec le DataFrame
Include.plot_histogram(input_norm)

# Assurez-vous que input_norm est un tableau 1D et le transformer en tableau 2D
input_data_scaled = scaler_X.transform(input_norm)
# Faire des prédictions
predictions = nn_model.predict(input_data_scaled)
#print('Prédictions :', predictions)
# Inverser la transformation
original_predictions = Include.inverse_transform_predictions(scaler_y, predictions)
#print('Prédictions :', original_predictions)

Include.plot_predictions(root_save, filename, original_predictions, integral)
