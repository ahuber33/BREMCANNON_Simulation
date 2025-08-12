# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:39:37 2025

@author: huber
"""

import sys
import uproot  # Pour lire les fichiers ROOT sans PyROOT
import awkward as ak  # uproot utilise awkward pour gérer les structures complexes
import os
import pandas as pd
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import keras
import GPy
from tqdm import trange
from scipy.signal import savgol_filter
shuffle_on= True

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, PReLU, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras.saving import register_keras_serializable

from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from scipy.stats.distributions import uniform, skewnorm

@register_keras_serializable()
class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
    

##############################################################################################    
##############################################################################################    
##############################################################################################    

def generer_spectre_uniforme(
    e_min=0,
    e_max=500000,  # en keV
    bin_size=100,   # en keV
    total_flux=1e7,
    poids=[0.35, 0.2, 0.1, 0.0, 0.35],  # expo, gauss, uniforme, piques, mix
    debug=True
):
    edges = np.arange(e_min, e_max + bin_size, bin_size)
    centres = (edges[:-1] + edges[1:]) / 2
    n = len(centres)

    choix = np.random.choice(["expo", "gauss", "uniforme", "piques", "mix"], p=poids)

    spectre = np.zeros_like(centres, dtype=float)

    if choix == "expo":
        #print(f"Forme choisie : {choix}")
        tau = 10 ** np.random.uniform(np.log10(1e2), np.log10(1e5))
        #print(f"Tau = {tau}")
        spectre = np.exp(-centres / tau)

    elif choix == "gauss":
        #print(f"Forme choisie : {choix}")
        mu = 10**np.random.uniform(np.log10(centres[0]), np.log10(centres[-1]))
        sigma = 10**np.random.uniform(np.log10(1e-3), np.log10(1)) * (mu)
        #print(f"mu = {mu}")
        #print(f"sigma = {sigma}")
        spectre = np.exp(-0.5 * ((centres - mu) / sigma)**2)

    elif choix == "uniforme":
        #print(f"Forme choisie : {choix}")
        e_start = np.random.uniform(centres[0], centres[-1]*0.7)
        e_stop = np.random.uniform(e_start, centres[-1])
        mask = (centres >= e_start) & (centres <= e_stop)
        #print(f"e start = {e_start}")
        #print(f"e stop = {e_stop}")
        spectre[mask] = 1.0

    elif choix == "piques":
        #print(f"Forme choisie : {choix}")
        nb = np.random.randint(1, 4)
        indices = np.random.choice(n, size=nb, replace=False)
        spectre[indices] = np.random.uniform(0.5, 1.0, size=nb)

    elif choix == "mix":
        #print(f"Forme choisie : {choix}")
        _, s1, _ = generer_spectre_uniforme(e_min, e_max, bin_size, 1, poids, debug=False)
        _, s2, _ = generer_spectre_uniforme(e_min, e_max, bin_size, 1, poids, debug=False)
        alpha = np.random.uniform(0.2, 0.8)
        spectre = alpha * s1 + (1 - alpha) * s2

    # Normalisation
    integrale = np.sum(spectre)
    if integrale > 0:
        spectre *= total_flux / integrale

    return centres, spectre, edges


##############################################################################################    
##############################################################################################    
##############################################################################################    



def rebin_spectre_personnalise(centres, spectre, edges_uniforme, binning_scheme):
    from collections import OrderedDict

    largeur_bin_initial = edges_uniforme[1] - edges_uniforme[0]

    # Construction des nouveaux bords
    new_edges = []
    for e_min, e_max, bin_width in binning_scheme:
        bins = list(np.arange(e_min, e_max, bin_width))
        if not bins or bins[-1] != e_max:
            bins.append(e_max)
        new_edges.extend(bins)

    new_edges = np.array(list(OrderedDict.fromkeys(new_edges)))

    new_spectre = np.zeros(len(new_edges) - 1)

    for i in range(len(new_edges) - 1):
        e1, e2 = new_edges[i], new_edges[i + 1]
        mask = (centres >= e1) & (centres < e2)
        # Multiplication par largeur bin initial pour avoir flux total par bin
        new_spectre[i] = np.sum(spectre[mask])

    new_centres = (new_edges[:-1] + new_edges[1:]) / 2

    largeur_bin_nouveau = new_edges[1:] - new_edges[:-1]
    integrale = np.sum(new_spectre)  # flux total, somme déjà intégrée sur les bins

    #print(f"Intégrale spectre rebinné : {integrale:.3e}")

    return new_edges, new_centres, new_spectre

    
    
##############################################################################################    
##############################################################################################    
##############################################################################################    
    
    
def tracer_spectres_binnés(spectre_incident, spectre_IP, binning_scheme):
    # Rebin inverse : reconvertir le spectre non uniforme vers binning uniforme
    centres_uniformes_recup, spectre_uniforme_recup = rebin_to_uniform(centres_non_uniforme, spectre_incident, edges_non_uniforme, edges_uniformes)
    
    # --- Affichage ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].step(edges_non_uniforme[:-1], spectre_incident, where='post', linewidth=2, label="Spectre rebinné non uniforme")
    axs[0].step(centres_uniformes_recup, spectre_uniforme_recup, where='mid', label="Spectre reconverti uniforme")
    
    axs[0].plot(centres_non_uniforme, spectre_incident, 'o', color='blue', markersize=3, label="Centres bin rebinnés")
    axs[0].plot(centres_uniformes_recup, spectre_uniforme_recup, 'o', color='cyan', markersize=2, label="Centres bin reconvertis uniforme")
    
    
    axs[0].set_xlabel("Énergie (keV)")
    axs[0].set_ylabel("Flux par bin")
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlim(10, 1e6)
    axs[0].set_ylim(1, 1e20)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Comparaison spectres : initial, rebinné, reconverti, reconverti lissé")
    
    axs[1].plot(range(1, len(spectre_IP) + 1), spectre_IP, marker='s')
    axs[1].set_xlabel("Numéro de bin IP")
    axs[1].set_ylabel("Signal IP")
    axs[1].set_title("Spectre IP simulé")
    axs[1].grid(True)
    
    plt.show()


##############################################################################################    
##############################################################################################    
##############################################################################################    



def rebin_to_uniform(centres_non_uniformes, spectre_non_uniforme, edges_non_uniformes, edges_uniformes):
    """
    Convertit un spectre d'un binning non uniforme vers un binning uniforme.

    Arguments :
        - centres_non_uniformes : centres des bins non uniformes
        - spectre_non_uniforme : flux total par bin (non uniforme)
        - edges_non_uniformes : bords des bins non uniformes
        - edges_uniformes : bords des bins uniformes souhaités

    Retour :
        - centres_uniformes : centres des bins uniformes
        - spectre_uniforme : flux total par bin uniforme
    """
    spectre_uniforme = np.zeros(len(edges_uniformes) - 1)

    for i in range(len(edges_non_uniformes) - 1):
        e1, e2 = edges_non_uniformes[i], edges_non_uniformes[i+1]
        flux_bin = spectre_non_uniforme[i]
        largeur_bin_non_uniforme = e2 - e1

        # Trouver les bins uniformes qui recouvrent ce bin non uniforme
        # Recherche des indices des bins uniformes qui intersectent [e1, e2]
        mask = (edges_uniformes[:-1] < e2) & (edges_uniformes[1:] > e1)
        bins_uniformes = np.where(mask)[0]

        for j in bins_uniformes:
            u1, u2 = edges_uniformes[j], edges_uniformes[j+1]

            # Calcul de l'intersection entre [e1,e2] et [u1,u2]
            overlap = max(0, min(e2, u2) - max(e1, u1))

            # Redistribution proportionnelle du flux selon la fraction d'énergie recouverte
            spectre_uniforme[j] += flux_bin * (overlap / largeur_bin_non_uniforme)

    centres_uniformes = (edges_uniformes[:-1] + edges_uniformes[1:]) / 2

    return centres_uniformes, spectre_uniforme




##############################################################################################    
##############################################################################################    
##############################################################################################    

def rebin_to_uniform_with_uncertainty(centres_non_uniformes, spectre_non_uniforme, err_non_uniforme,
                                      edges_non_uniformes, edges_uniformes):
    """
    Rebinne un spectre avec incertitudes d'un binning non uniforme à uniforme.

    Arguments :
        - centres_non_uniformes : centres des bins non uniformes
        - spectre_non_uniforme : flux total par bin
        - err_non_uniforme : incertitudes sur chaque bin (écart-type)
        - edges_non_uniformes : bords des bins non uniformes
        - edges_uniformes : bords des bins uniformes

    Retour :
        - centres_uniformes : centres des bins uniformes
        - spectre_uniforme : flux total par bin uniforme
        - err_uniforme : incertitude (écart-type) sur chaque bin uniforme
    """
    n_uniform = len(edges_uniformes) - 1
    spectre_uniforme = np.zeros(n_uniform)
    err2_uniforme = np.zeros(n_uniform)  # on stocke le carré des erreurs

    for i in range(len(edges_non_uniformes) - 1):
        e1, e2 = edges_non_uniformes[i], edges_non_uniformes[i+1]
        flux_bin = spectre_non_uniforme[i]
        err_bin = err_non_uniforme[i]
        largeur_bin_non_uniforme = e2 - e1

        # Trouver les bins uniformes qui recouvrent ce bin non uniforme
        mask = (edges_uniformes[:-1] < e2) & (edges_uniformes[1:] > e1)
        bins_uniformes = np.where(mask)[0]

        for j in bins_uniformes:
            u1, u2 = edges_uniformes[j], edges_uniformes[j+1]

            # Calcul de l'intersection entre [e1,e2] et [u1,u2]
            overlap = max(0, min(e2, u2) - max(e1, u1))
            weight = overlap / largeur_bin_non_uniforme

            # Redistribuer le flux et l'incertitude proportionnellement
            spectre_uniforme[j] += flux_bin * weight
            err2_uniforme[j] += (err_bin * weight) ** 2  # somme quadratique

    centres_uniformes = (edges_uniformes[:-1] + edges_uniformes[1:]) / 2
    err_uniforme = np.sqrt(err2_uniforme)

    return centres_uniformes, spectre_uniforme, err_uniforme




##############################################################################################    
##############################################################################################    
##############################################################################################    

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


##############################################################################################    
##############################################################################################    
##############################################################################################    

def ML_Estimation_GP(path, gp_model, scaler_X, scaler_Y, spectre_IP) :
    df1 = pd.read_pickle(path + 'scalers/Dummy.pickle')
    
    # Refaire la sélection des colonnes features
    x = df1[["PSL_IP_1", "PSL_IP_2", "PSL_IP_3", "PSL_IP_4", "PSL_IP_5",
         "PSL_IP_6", "PSL_IP_7", "PSL_IP_8", "PSL_IP_9", "PSL_IP_10",
         "PSL_IP_11", "PSL_IP_12", "PSL_IP_13", "PSL_IP_14", "PSL_IP_15"]]
    
    spectre_IPb = spectre_IP[np.newaxis, :]# Normalisation ligne par ligne

    sum_spectre = spectre_IPb.sum(axis=1, keepdims=True) + 1e-20
    spectre_IPb_norm = spectre_IPb / sum_spectre
    spectre_IPlog = np.log10(spectre_IPb_norm + 1e-20)
    
    input_data_scaled_gpy = scaler_X.transform(pd.DataFrame(spectre_IPlog, columns=x.columns))

    # Prédiction GP
    predictions_gauss, sigma_predictions_gauss = gp_model.predict(input_data_scaled_gpy, return_std=True)
    
    # Inverse scaling sur les prédictions GP (espace log10)
    predictions_gauss_log = inverse_transform_predictions(scaler_Y, predictions_gauss)
    sigma_predictions_gauss_log = sigma_predictions_gauss * (scaler_Y.data_range_)
    
    # Retour espace physique
    original_predictions_gauss = 10 ** predictions_gauss_log
    
    # Gestion des incertitudes GP
    pred_upper_log = predictions_gauss_log + sigma_predictions_gauss_log
    pred_lower_log = predictions_gauss_log - sigma_predictions_gauss_log
    pred_upper = 10 ** pred_upper_log
    pred_lower = 10 ** pred_lower_log
    original_sigma_predictions_gauss = pred_upper - original_predictions_gauss

    # Récupérer le facteur scalaire
    sum_spectre_scalar = sum_spectre.squeeze()  # transformera shape (1,1) → scalaire

    # Dé-normaliser chaque prédiction
    original_predictions_gauss *= sum_spectre_scalar
    
    # Dé-normaliser les incertitudes associées (sigma) également :
    original_sigma_predictions_gauss *= sum_spectre_scalar

    epsilon = 1e-12  # ou autre petite valeur adaptée à ton domaine
    sigma_log = original_sigma_predictions_gauss[0] / ((original_predictions_gauss[0] - epsilon) * np.log(10))
    sum_sigma_log = np.sum(sigma_log[sigma_log>0])
    print(sum_sigma_log)

    return original_predictions_gauss, original_sigma_predictions_gauss, sum_sigma_log


##############################################################################################    
##############################################################################################    
##############################################################################################    

def ML_Estimation_RF(path, rf_model, scaler_X, scaler_Y, spectre_IP) :
    df1 = pd.read_pickle(path + 'scalers/Dummy.pickle')
    
    # Refaire la sélection des colonnes features
    x = df1[["PSL_IP_1", "PSL_IP_2", "PSL_IP_3", "PSL_IP_4", "PSL_IP_5",
         "PSL_IP_6", "PSL_IP_7", "PSL_IP_8", "PSL_IP_9", "PSL_IP_10",
         "PSL_IP_11", "PSL_IP_12", "PSL_IP_13", "PSL_IP_14", "PSL_IP_15"]]
    
    spectre_IPb = spectre_IP[np.newaxis, :]# Normalisation ligne par ligne

    sum_spectre = spectre_IPb.sum(axis=1, keepdims=True) + 1e-20
    spectre_IPb_norm = spectre_IPb / sum_spectre
    spectre_IPlog = np.log10(spectre_IPb_norm + 1e-20)

    input_data_scaled_rf = scaler_X.transform(pd.DataFrame(spectre_IPlog, columns=x.columns))

    # Prédiction Random Forest
    # Récupération des prédictions de chaque arbre (dans l’espace standardisé log10)
    all_simulations_rf = np.array([
        tree.predict(input_data_scaled_rf) for tree in rf_model.estimators_
    ])  # shape: (n_trees, n_samples)
    
    # Appliquer l'inverse scaling dans l'espace log10
    simulations_log_rf = np.array([
        inverse_transform_predictions(scaler_Y, sim.reshape(1, -1)).squeeze()
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

    # Récupérer le facteur scalaire
    sum_spectre_scalar = sum_spectre.squeeze()  # transformera shape (1,1) → scalaire
    
    # Dé-normaliser chaque prédiction
    original_predictions_rf *= sum_spectre_scalar
    
    # Dé-normaliser les incertitudes associées (sigma) également :
    original_sigma_predictions_rf *= sum_spectre_scalar

    epsilon = 1e-12  # ou autre petite valeur adaptée à ton domaine
    sigma_log = original_sigma_predictions_rf / ((original_predictions_rf - epsilon) * np.log(10))
    sum_sigma_log = np.sum(sigma_log[sigma_log>0])
    print(sum_sigma_log)

    return original_predictions_rf, original_sigma_predictions_rf, sum_sigma_log


##############################################################################################    
##############################################################################################    
##############################################################################################    

def plot_predictions_BEST(centres_uniforme, predictions, sigma_predictions, GP_bool, label, color):
    centres, dummy, edges = generer_spectre_uniforme(bin_size=100, total_flux=1)
    # Rebinning
    new_edges, new_centres, new_dummy = rebin_spectre_personnalise(centres_uniforme, dummy, edges_uniforme, binning_scheme)

    bin_labels_ext = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 
              1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500,
              12500, 17500, 22500, 27500, 32500, 37500, 42500, 47500, 52500, 57500, 62500, 67500, 72500, 77500, 82500, 87500, 92500, 97500,
              125000, 175000, 225000, 275000, 325000, 375000, 425000, 475000]

    bin_labels_ext_err = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 
              500, 500, 500, 500, 500, 500, 500, 500, 500,
              2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,
              25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000]
    
    """
    Plot the predictions as a histogram and display the ratio between predicted and actual values.
    Also computes and displays R² values.
    """
    bin_x_errors = 100/2

    # Rebinning
    if GP_bool == True :
        centres_uniforme, spectre_uniforme, err_uniforme = rebin_to_uniform_with_uncertainty(new_centres, predictions[0], sigma_predictions[0],
                                      new_edges, edges_uniforme)

    else :
        centres_uniforme, spectre_uniforme, err_uniforme = rebin_to_uniform_with_uncertainty(new_centres, predictions, sigma_predictions,
                                      new_edges, edges_uniforme)
    

    # --- Affichage ---
    plt.figure(figsize=(12, 6))
    
    plt.errorbar(centres_uniforme, spectre_uniforme, xerr=bin_x_errors, yerr=err_uniforme, fmt='.', capsize=1,
                    color=color, alpha=0.8, linewidth=0.01, 
                    label = label)

    plt.fill_between(centres_uniforme, spectre_uniforme - err_uniforme, spectre_uniforme + err_uniforme, color=color, alpha=0.25)
    plt.legend()

    plt.xlabel("Énergie (keV)")
    plt.ylabel("Flux par bin")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, 1e6)
    plt.ylim(1, 10*np.max(spectre_uniforme))
    plt.legend()
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()

    return centres_uniforme, spectre_uniforme, bin_x_errors, err_uniforme

##############################################################################################    
##############################################################################################    
##############################################################################################    


def plot_predictions_Other(centres_uniforme, predictions_1, sigma_predictions_1, var_1, GP_bool_1, label_1, color_1,
                             predictions_2, sigma_predictions_2, var_2, GP_bool_2, label_2, color_2,
                             predictions_3, sigma_predictions_3, var_3, GP_bool_3, label_3, color_3):
    """
    Affiche 3 spectres dans des subplots triés par variance croissante (le meilleur en haut).
    Chaque spectre est tracé avec ses barres d'erreur.
    """

    # --- Créer une liste des spectres avec toutes leurs infos ---
    spectres = [
        {"pred": predictions_1, "sigma": sigma_predictions_1, "var": var_1, "gp": GP_bool_1, "label": label_1, "color": color_1},
        {"pred": predictions_2, "sigma": sigma_predictions_2, "var": var_2, "gp": GP_bool_2, "label": label_2, "color": color_2},
        {"pred": predictions_3, "sigma": sigma_predictions_3, "var": var_3, "gp": GP_bool_3, "label": label_3, "color": color_3},
    ]

    # --- Trier par variance croissante ---
    spectres_sorted = sorted(spectres, key=lambda s: s["var"])
    bin_x_errors = 100/2

    # --- Génération des spectres d'énergie pour l'axe X ---
    new_edges, new_centres, new_dummy = rebin_spectre_personnalise(centres_uniforme, dummy, edges_uniforme, binning_scheme)

    # --- Créer la figure avec 3 subplots verticaux ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    for idx, s in enumerate(spectres_sorted):
        ax = axs[idx]

        # --- Rebinning selon le type de modèle ---
        if s["gp"]:
            centres_uniforme, spectre_uniforme, err_uniforme = rebin_to_uniform_with_uncertainty(
                new_centres, s["pred"][0], s["sigma"][0], new_edges, edges_uniforme)
        else:
            centres_uniforme, spectre_uniforme, err_uniforme = rebin_to_uniform_with_uncertainty(
                new_centres, s["pred"], s["sigma"], new_edges, edges_uniforme)

        # --- Tracer le spectre ---
        ax.errorbar(
            centres_uniforme, spectre_uniforme, xerr=50, yerr=err_uniforme, fmt='.', capsize=1,
            color=s["color"], alpha=0.8, linewidth=0.01, label=f'{s["label"]}'
        )
        ax.fill_between(
            centres_uniforme, spectre_uniforme - err_uniforme, spectre_uniforme + err_uniforme,
            color=s["color"], alpha=0.25
        )

        # Mise en forme du subplot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10, 1e6)
        ax.set_ylim(1, 10*np.max(spectre_uniforme))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()
        ax.set_ylabel("Flux par bin")
        if idx == 2:
            ax.set_xlabel("Énergie (keV)")

    plt.tight_layout()
    plt.show()



##############################################################################################    
##############################################################################################    
##############################################################################################    


def plot_predictions_ALL(centres_uniforme, predictions_1, sigma_predictions_1, var_1, GP_bool_1, label_1, color_1,
                         predictions_2, sigma_predictions_2, var_2, GP_bool_2, label_2, color_2,
                         predictions_3, sigma_predictions_3, var_3, GP_bool_3, label_3, color_3,
                         predictions_4, sigma_predictions_4, var_4, GP_bool_4, label_4, color_4):
    """
    Affiche 3 spectres dans des subplots triés par variance croissante (le meilleur en haut).
    Chaque spectre est tracé avec ses barres d'erreur.
    """

    # --- Créer une liste des spectres avec toutes leurs infos ---
    spectres = [
        {"pred": predictions_1, "sigma": sigma_predictions_1, "var": var_1, "gp": GP_bool_1, "label": label_1, "color": color_1},
        {"pred": predictions_2, "sigma": sigma_predictions_2, "var": var_2, "gp": GP_bool_2, "label": label_2, "color": color_2},
        {"pred": predictions_3, "sigma": sigma_predictions_3, "var": var_3, "gp": GP_bool_3, "label": label_3, "color": color_3},
        {"pred": predictions_4, "sigma": sigma_predictions_4, "var": var_4, "gp": GP_bool_4, "label": label_4, "color": color_4},
    ]

    # --- Trier par variance croissante ---
    spectres_sorted = sorted(spectres, key=lambda s: s["var"])
    bin_x_errors = 100/2

    # --- Génération des spectres d'énergie pour l'axe X ---
    new_edges, new_centres, new_dummy = rebin_spectre_personnalise(centres_uniforme, dummy, edges_uniforme, binning_scheme)

    # --- Affichage ---
    plt.figure(figsize=(12, 6))

    for spec in spectres_sorted:
        # Rebinning en fonction du type de modèle (GP ou pas)
        if spec["gp"]:
            centres_uniforme, spectre_uniforme, err_uniforme = rebin_to_uniform_with_uncertainty(
                new_centres, spec["pred"][0], spec["sigma"][0], new_edges, edges_uniforme
            )
        else:
            centres_uniforme, spectre_uniforme, err_uniforme = rebin_to_uniform_with_uncertainty(
                new_centres, spec["pred"], spec["sigma"], new_edges, edges_uniforme
            )

        # Affichage avec barres d’erreur
        plt.errorbar(
            centres_uniforme, spectre_uniforme,
            xerr=bin_x_errors, yerr=err_uniforme,
            fmt='.', capsize=1,
            color=spec["color"], alpha=0.8, linewidth=0.01,
            label=f"{spec['label']}"
        )
        plt.fill_between(
            centres_uniforme,
            spectre_uniforme - err_uniforme,
            spectre_uniforme + err_uniforme,
            color=spec["color"], alpha=0.25
        )

    plt.xlabel("Énergie (keV)")
    plt.ylabel("Flux par bin")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, 1e6)
    plt.ylim(1, 1e20)
    plt.legend()
    plt.title("Superposition de toutes les solutions")
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()



##############################################################################################    
##############################################################################################    
##############################################################################################    

def Load_ShotFile(path, filename):
    # Chemin vers votre fichier texte
    file_path = path + filename

    # Ouvrir le fichier en mode lecture
    with open(file_path, 'r') as file:
        # Lire la première ligne
        first_line = file.readline().strip()
        # Lire le reste des lignes
        data_lines = file.readlines()

    # Extraire les métadonnées de la première ligne
    metadata = first_line.split(',')
    energy = metadata[0].split(' ')[1]
    pression = metadata[1].split(' ')[2]
    yyyy = metadata[2].split(' ')[2]

    # Extraire les valeurs numériques des lignes suivantes de façon robuste
    spectre_IP = []
    for line in data_lines:
        cleaned_line = line.strip()
        try:
            value = float(cleaned_line)
            spectre_IP.append(value)
        except ValueError:
            # Ligne non numérique => on l’ignore
            continue

    spectre_IP = np.array(spectre_IP)
    #sigma_spectre_IP = np.sqrt(spectre_IP)
    sigma_spectre_IP = 0.1*(spectre_IP)

    # Afficher les résultats
    print("Métadonnées:")
    print(f"Energy: {energy}, Pression: {pression}, yyyy: {yyyy}")
    print("Valeurs:")
    print(spectre_IP)

    return spectre_IP, sigma_spectre_IP, metadata



##############################################################################################    
##############################################################################################    
##############################################################################################    

def Estimation_Best_ML(path, centres_uniforme):
    rf_predi_31, rf_sigma_predi_31, var_rf_31 = ML_Estimation_RF(path, rf_model_31, scaler_X_31, scaler_Y_31, spectre_IP)
    rf_predi_35, rf_sigma_predi_35, var_rf_35 = ML_Estimation_RF(path, rf_model_35, scaler_X_35, scaler_Y_35, spectre_IP)
    
    gp_predi_31, gp_sigma_predi_31, var_gp_31 = ML_Estimation_GP(path, gp_model_31, scaler_X_31, scaler_Y_31, spectre_IP)
    gp_predi_35, gp_sigma_predi_35, var_gp_35 = ML_Estimation_GP(path, gp_model_35, scaler_X_35, scaler_Y_35, spectre_IP)
    
    rf_31 = False
    rf_35 = False
    gp_31 = False
    gp_35 = False
    
    if var_rf_31 <8 :
        rf_31 =True
    
    if var_rf_35 <25 :
        rf_35 =True
    
    if var_gp_31 < 100 and var_gp_31 < var_gp_35:
        gp_31 =True
    
    if var_gp_35 <10000 and var_gp_35 < var_gp_31:
        gp_35 =True
    
    print(rf_31)
    print(rf_35)
    print(gp_31)
    print(gp_35)
    
    if rf_31 ==True :
        print("Best Solution with RF31")
        centre_best, spectre_best, err_x, err_y = plot_predictions_BEST(centres_uniforme, rf_predi_31, rf_sigma_predi_31, False, "RF Model 31: " + str(round(var_rf_31,1)), "black")
        plot_predictions_Other(centres_uniformes, rf_predi_35, rf_sigma_predi_35, var_rf_35, False, "RF Model 35: " + str(round(var_rf_35,1)), "green",
                               gp_predi_31, gp_sigma_predi_31, var_gp_31, True, "GP Model 31: " + str(round(var_gp_31,1)), "red",
                               gp_predi_35, gp_sigma_predi_35, var_gp_35, True, "GP Model 35: " + str(round(var_gp_35,1)), "darkorange")
        
    if rf_31 ==False and rf_35==True :
        print("Best Solution with RF35")    
        centre_best, spectre_best, err_x, err_y = plot_predictions_BEST(centres_uniforme, rf_predi_35, rf_sigma_predi_35, False, "RF Model 35: " + str(round(var_rf_35,1)), "black")
        plot_predictions_Other(centres_uniforme, rf_predi_31, rf_sigma_predi_31, var_rf_31, False, "RF Model 31: " + str(round(var_rf_31,1)), "lime",
                               gp_predi_31, gp_sigma_predi_31, var_gp_31, True, "GP Model 31: " + str(round(var_gp_31,1)), "red",
                               gp_predi_35, gp_sigma_predi_35, var_gp_35, True, "GP Model 35: " + str(round(var_gp_35,1)), "darkorange")
        
    
    if rf_31 ==False and rf_35==False and gp_31 ==True :
        print("Best Solution with GP31")        
        centre_best, spectre_best, err_x, err_y = plot_predictions_BEST(centres_uniforme, gp_predi_31, gp_sigma_predi_31, True, "GP Model 31: " + str(round(var_gp_31,1)), "black")
        plot_predictions_Other(centres_uniforme, rf_predi_31, rf_sigma_predi_31, var_rf_31, False, "RF Model 31: " + str(round(var_rf_31,1)), "lime",
                               rf_predi_35, rf_sigma_predi_35, var_rf_35, False, "RF Model 35: " + str(round(var_rf_35,1)), "green",
                               gp_predi_35, gp_sigma_predi_35, var_gp_35, True, "GP Model 35: " + str(round(var_gp_35,1)), "darkorange")
        
    
    if rf_31 ==False and rf_35==False and gp_31 ==False and gp_35 ==True :
        print("Best Solution with GP35")            
        centre_best, spectre_best, err_x, err_y = plot_predictions_BEST(centres_uniforme, gp_predi_35, gp_sigma_predi_35, True, "GP Model 35: " + str(round(var_gp_35,1)), "black")
        plot_predictions_Other(centres_uniforme, rf_predi_31, rf_sigma_predi_31, var_rf_31, False, "RF Model 31: " + str(round(var_rf_31,1)), "lime",
                               rf_predi_35, rf_sigma_predi_35, var_rf_35, False, "RF Model 35: " + str(round(var_rf_35,1)), "green",
                               gp_predi_31, gp_sigma_predi_31, var_gp_31, True, "GP Model 31: " + str(round(var_gp_31,1)), "red")

    if rf_31 ==False and rf_35==False and gp_31 ==False and gp_35 ==False :
        print("Best Solution default RF31")            
        centre_best, spectre_best, err_x, err_y = plot_predictions_BEST(centres_uniforme, rf_predi_31, rf_sigma_predi_31, False, "RF Model 31: " + str(round(var_rf_31,1)), "black")
        plot_predictions_Other(centres_uniforme, rf_predi_35, rf_sigma_predi_35, var_rf_35, False, "RF Model 35: " + str(round(var_rf_35,1)), "green",
                               gp_predi_31, gp_sigma_predi_31, var_gp_31, True, "GP Model 31: " + str(round(var_gp_31,1)), "red",
                               gp_predi_35, gp_sigma_predi_35, var_gp_35, True, "GP Model 35: " + str(round(var_gp_35,1)), "darkorange")
        
    
    plot_predictions_ALL(centres_uniforme, rf_predi_31, rf_sigma_predi_31, var_rf_31, False, "RF Model 31: " + str(round(var_rf_31,1)), "lime",
                           rf_predi_35, rf_sigma_predi_35, var_rf_35, False, "RF Model 35: " + str(round(var_rf_35,1)), "green",
                           gp_predi_31, gp_sigma_predi_31, var_gp_31, True, "GP Model 31: " + str(round(var_gp_31,1)), "red",
                           gp_predi_35, gp_sigma_predi_35, var_gp_35, True, "GP Model 35: " + str(round(var_gp_35,1)), "darkorange")    

    return centre_best, spectre_best, err_x, err_y


##############################################################################################    
##############################################################################################    
##############################################################################################    

def Calcul_Uncertainty_single_model(m_single):
    # Paramètres du fit (issus de Minuit)
    A_fit = m_single.values["A"]
    E0_fit = m_single.values["E0"]
    # Récupération de la covariance et conversion en matrice numpy
    cov = m_single.covariance  # dictionnaire 2D
    cov_np = np.array([
        [cov["A", "A"],  cov["A", "E0"]],
        [cov["E0", "A"], cov["E0", "E0"]]
    ])
    
    E = centres_Minuit
    
    # Dérivées du modèle par rapport à A et E0
    df_dA = np.exp(-E / E0_fit)
    df_dE0 = A_fit * df_dA * (E / E0_fit**2)
    
    # Gradient J (shape: 45 x 2)
    J = np.vstack((df_dA, df_dE0)).T
    
    # Propagation des incertitudes avec la covariance
    sigma2_model = np.einsum("ij,jk,ik->i", J, cov, J)
    sigma_model = np.sqrt(sigma2_model)

    return sigma_model


##############################################################################################    
##############################################################################################    
##############################################################################################    

def Calcul_Uncertainty_dual_model(m_dual):
    # Paramètres du fit
    A0 = m_dual.values["A0"]
    E01 = m_dual.values["E01"]
    A1 = m_dual.values["A1"]
    E02 = m_dual.values["E02"]

    E = centres_Minuit
    
    # Dérivées analytiques
    df_dA0 = np.exp(-E / E01)
    df_dE01 = A0 * df_dA0 * (E / E01**2)
    df_dA1 = np.exp(-E / E02)
    df_dE02 = A1 * df_dA1 * (E / E02**2)
    
    # Stack des dérivées (shape: 45 x 4)
    J = np.vstack((df_dA0, df_dE01, df_dA1, df_dE02)).T
    
    # Récupérer la matrice de covariance 4x4
    cov_dict = m_dual.covariance
    param_order = ["A0", "E01", "A1", "E02"]
    cov_np = np.array([[cov_dict[p1, p2] for p2 in param_order] for p1 in param_order])
    
    # Propagation : σ² = Jᵢ @ Cov @ Jᵢᵗ
    sigma2_model = np.einsum("ij,jk,ik->i", J, cov_np, J)
    sigma_model = np.sqrt(sigma2_model)

    return sigma_model

##############################################################################################    
##############################################################################################    
##############################################################################################    

def rebin_Minuit_function(centres, spectre, sigma, edges_uniforme, binning_scheme):
    from collections import OrderedDict
    import numpy as np

    largeur_bin_initial = edges_uniforme[1] - edges_uniforme[0]

    # Construction des nouveaux bords
    new_edges = []
    for e_min, e_max, bin_width in binning_scheme:
        bins = list(np.arange(e_min, e_max, bin_width))
        if not bins or bins[-1] != e_max:
            bins.append(e_max)
        new_edges.extend(bins)

    # Supprimer les doublons éventuels tout en conservant l'ordre
    new_edges = np.array(list(OrderedDict.fromkeys(new_edges)))

    # Initialisation
    new_spectre = np.zeros(len(new_edges) - 1)
    new_sigma2 = np.zeros(len(new_edges) - 1)  # variances

    for i in range(len(new_edges) - 1):
        e1, e2 = new_edges[i], new_edges[i + 1]
        mask = (centres >= e1) & (centres < e2)

        # Rebin du spectre (somme des contenus dans les bins initiaux)
        new_spectre[i] = np.sum(spectre[mask])

        # Propagation des erreurs (somme des variances)
        new_sigma2[i] = np.sum(sigma[mask]**2)

    # Calcul des nouveaux centres et largeurs de bins
    new_centres = (new_edges[:-1] + new_edges[1:]) / 2
    new_sigma = np.sqrt(new_sigma2)

    return new_edges, new_centres, new_spectre, new_sigma

##############################################################################################    
##############################################################################################    
##############################################################################################    

def spectrum(E, A, E0):
    return A * np.exp(-E / E0)  # shape (500,)

def dual_spectrum(E, A0, E01, A1, E02):

    return A0*np.exp(-E / E01) + A1 * np.exp(-E / E02)

def triple_spectrum(E, A0, E01, A1, E02, A2, mu, sigma):
    exp1 = A0 * np.exp(-E / E01)
    exp2 = A1 * np.exp(-E / E02)
    gaussian = A2 * np.exp(-0.5 * ((E - mu) / sigma) ** 2)
    return exp1 + exp2 + gaussian    


def model(A, E0):
    S_E = spectrum(centres_Minuit, A, E0)
    
    # Rebinning
    new_edges, new_centres, S_E = rebin_spectre_personnalise(centres_Minuit, S_E, edges_Minuit, binning_scheme)
    #print(S_E)
    S_IP = Matrice @ S_E  # shape (15,)

    return S_IP

    
def model_plot(m_single):
    A = m_single.values["A"]
    E0 = m_single.values["E0"]
    S_E = spectrum(centres_Minuit, A, E0)
    #print(S_E)
    sigma_model = Calcul_Uncertainty_single_model(m_single)
    
    # Rebinning
    new_edges, new_centres, S_E, S_E_sigma = rebin_Minuit_function(centres_Minuit, S_E, sigma_model, edges_Minuit, binning_scheme)
    #print(S_E)
    S_IP = Matrice @ S_E  # shape (15,)
    # Calcul des variances du modèle
    sigma2_ip_model = (Matrice**2) @ (S_E_sigma**2)
    
    # Et les écarts-types
    sig_IP = np.sqrt(sigma2_ip_model)
    #print(len(S_E))
    return S_IP, sig_IP

def dual_model(A0, E01, A1, E02):
    S_E = dual_spectrum(centres_Minuit, A0, E01, A1, E02)
    
    # Rebinning
    new_edges, new_centres, S_E = rebin_spectre_personnalise(centres_Minuit, S_E, edges_Minuit, binning_scheme)
    
    S_IP = Matrice @ S_E  # shape (15,)

    return S_IP    

def dual_model_plot(m_dual):
    A0 = m_dual.values["A0"]
    E01 = m_dual.values["E01"]
    A1 = m_dual.values["A1"]
    E02 = m_dual.values["E02"]
    S_E = dual_spectrum(centres_Minuit, A0, E01, A1, E02)
    #print(S_E)
    sigma_model_dual = Calcul_Uncertainty_dual_model(m_dual)
    
    # Rebinning
    new_edges, new_centres, S_E, S_E_sigma = rebin_Minuit_function(centres_Minuit, S_E, sigma_model_dual, edges_Minuit, binning_scheme)
    #print(S_E)
    S_IP = Matrice @ S_E  # shape (15,)
    # Calcul des variances du modèle
    sigma2_ip_model = (Matrice**2) @ (S_E_sigma**2)
    
    # Et les écarts-types
    sig_IP = np.sqrt(sigma2_ip_model)
    #print(len(S_E))
    return S_IP, sig_IP    

def chi2(A, E0):
    model_IP = model(A, E0)
    return np.sum(((spectre_IP - model_IP) / sigma_spectre_IP) ** 2)

def dual_chi2(A0, E01, A1, E02):
    model_IP = dual_model(A0, E01, A1, E02)
    return np.sum(((spectre_IP - model_IP) / sigma_spectre_IP) ** 2)


    
##############################################################################################    
##############################################################################################    
##############################################################################################    

def Fit_Minuit():
    from iminuit import Minuit
    
    # Ajustement à une température
    #print(np.max(spectre_best))
    m_single = Minuit(chi2, A=np.max(spectre_best), E0=1000.0)
    m_single.limits["A"] = (np.max(spectre_best)*1e-3, np.max(spectre_best)*1e3)
    m_single.limits["E0"] = (0.1, 100000)
    m_single.migrad()
    print("Single Temperature Fit:")
    print("Best parameters:", m_single.values)
    print("Errors:", m_single.errors)
    print("Minimum chi-squared:", m_single.fval)
    
    # Ajustement à deux températures
    m_dual = Minuit(dual_chi2, A0=m_single.values["A"], E01=m_single.values["E0"], A1=m_single.values["A"], E02=10000.0)
    m_dual.limits["A0"] = (np.max(spectre_best)*1e-6, np.max(spectre_best)*1e6)
    m_dual.limits["E01"] = (0.1, 10000)
    m_dual.limits["A1"] = (np.max(spectre_best)*1e-6, np.max(spectre_best)*1e6)
    m_dual.limits["E02"] = (10000, 100000)
    m_dual.migrad()
    print("\nDual Temperature Fit:")
    print("Best parameters:", m_dual.values)
    print("Errors:", m_dual.errors)
    print("Minimum chi-squared:", m_dual.fval)

    return m_single, m_dual

##############################################################################################    
##############################################################################################    
##############################################################################################    

def chi2_complet(best_model_IP, sig_best_model_IP):
    variance_totale = sigma_spectre_IP**2 + sig_best_model_IP**2
    chi2 = np.sum((spectre_IP - best_model_IP)**2 / variance_totale)
    return chi2


##############################################################################################    
##############################################################################################    
##############################################################################################    


def Plot_Fit_Minuit_IP(path, filename, metadata):
    new_edges, new_centres, S_E, S_E_sigma = rebin_Minuit_function(centre_best, spectre_best, err_y, edges_Minuit, binning_scheme)
    S_IP_ML = Matrice @ S_E  # shape (15,)
    # Calcul des variances du modèle
    sigma2_ip_model = (Matrice**2) @ (S_E_sigma**2)
    sig_IP_ML = np.sqrt(sigma2_ip_model)
    
    best_model_IP, sig_best_model_IP = model_plot(m_single)
    best_model_IP_dual, sig_best_model_IP_dual = dual_model_plot(m_dual)
    

    ip_indices = range(1, 16)  # 1 à 15 inclus

    chi2_mono = chi2_complet(best_model_IP, sig_best_model_IP)
    chi2_dual = chi2_complet(best_model_IP_dual, sig_best_model_IP_dual)
    chi2_ML = chi2_complet(S_IP_ML, sig_IP_ML)
    chi2_ML_P = chi2_complet(PSL_ML_P, sigma_PSL_ML_P)
    chi2_P = chi2_complet(PSL_P, sigma_PSL_P)
    chi2_dual_P = chi2_complet(PSL_dual_P, sigma_PSL_dual_P)

    plt.errorbar(ip_indices, spectre_IP, yerr=sigma_spectre_IP, fmt="o", label="Données", color="black")
    plt.plot(ip_indices, best_model_IP, label=f"Fit 1T : Chi2 fit={m_single.fval:.1f} ; Chi2 global={chi2_mono:.1f}", color="red")
    plt.fill_between(ip_indices, best_model_IP - sig_best_model_IP, best_model_IP + sig_best_model_IP,
             alpha=0.3, color="red")
    plt.plot(ip_indices, best_model_IP_dual, label=f"Fit 2T : Chi2 fit={m_dual.fval:.1f} ; Chi2 global={chi2_dual:.1f}", color="blue")
    plt.fill_between(ip_indices, best_model_IP_dual - sig_best_model_IP_dual, best_model_IP_dual + sig_best_model_IP_dual,
             alpha=0.3, color="blue")
    plt.plot(ip_indices, S_IP_ML, label=f"ML model : Chi2 global={chi2_ML:.1f}", color="cyan")
    plt.fill_between(ip_indices, S_IP_ML - sig_IP_ML, S_IP_ML + sig_IP_ML,
             alpha=0.3, color="cyan")
    plt.plot(ip_indices, PSL_ML_P, label=f"Perturbative Algo from ML model : Chi2 global={chi2_ML_P:.1f}", color="orange")
    plt.fill_between(ip_indices, PSL_ML_P - sigma_PSL_ML_P, PSL_ML_P + sigma_PSL_ML_P,
             alpha=0.3, color="orange")
    plt.plot(ip_indices, PSL_P, label=f"Perturbative Algo from Uniform : Chi2 global={chi2_P:.1f}", color="gray")
    plt.fill_between(ip_indices, PSL_P - sigma_PSL_P, PSL_P + sigma_PSL_P,
             alpha=0.2, color="gray")
    plt.plot(ip_indices, PSL_dual_P, label=f"Perturbative Algo from 2T Fit : Chi2 global={chi2_dual_P:.1f}", color="green")
    plt.fill_between(ip_indices, PSL_dual_P - sigma_PSL_dual_P, PSL_dual_P + sigma_PSL_dual_P,
             alpha=0.2, color="green")
    #plt.plot(ip_indices, best_model_IP_triple, label="Fit 2 Température + Gauss", color="cyan")
    
    plt.xlabel("IP #")
    plt.ylabel("PSL/mm²")
    #plt.yscale("log")
    
    # Mettre des ticks majeurs tous les 1 sur X
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    
    # Afficher la grille alignée sur les ticks majeurs
    plt.grid(True, which="major", axis="x", ls="--")
    plt.grid(True, which="both", axis="y", ls="--")  # optionnel : grille sur Y aussi
    
    plt.legend()
    plt.title(metadata)
    plt.tight_layout()
    
    # Chemin complet du fichier (nom au choix)
    filename = os.path.join(path, filename+"_Minuit_IP.png")

    # Sauvegarde
    plt.savefig(filename)
    
    plt.show()
    
    
##############################################################################################    
##############################################################################################    
##############################################################################################    

def Plot_Fit_Minuit_Spectre(path, filename, metadata):
    best_A = m_single.values["A"]
    best_E0 = m_single.values["E0"]
    best_A0 = m_dual.values["A0"]
    best_E01 = m_dual.values["E01"]
    best_A1 = m_dual.values["A1"]
    best_E02 = m_dual.values["E02"]
    
    print("Fit 1 Température :")
    print(rf"Best A = {best_A:.2e}")
    print(rf"Best E0 = {best_E0:.2e}")
    print(rf"Best Chi2 = {m_single.fval:.2e}")
    print("Fit 2 Températures :")
    print(rf"Best A0 = {best_A0:.2e}")
    print(rf"Best E01 = {best_E01:.2e}")
    print(rf"Best A1 = {best_A1:.2e}")
    print(rf"Best E02 = {best_E02:.2e}")
    print(rf"Best Chi2 = {m_dual.fval:.2e}")

    sigma_model = Calcul_Uncertainty_single_model(m_single)
    sigma_model_dual = Calcul_Uncertainty_dual_model(m_dual)
    
    S_E_best = spectrum(E, best_A, best_E0)  # shape (500,)
    S_E_best_dual = dual_spectrum(E, best_A0, best_E01, best_A1, best_E02)  # shape (500,)
    
    plt.figure(figsize=(7,5))
    plt.plot(E, S_E_best, label=f"Fit : A={best_A:.1e}, E0={best_E0:.2f} keV, Chi2={m_single.fval:.1f}", color="red")
    plt.fill_between(E, S_E_best - sigma_model, S_E_best + sigma_model,
                 alpha=0.3, color="red")
    plt.plot(E, S_E_best_dual, label=f"Fit : A0={best_A0:.1e}, E01={best_E01:.2f} keV, A1={best_A1:.1e}, E02={best_E02:.2f} keV, Chi2={m_dual.fval:.1f}", color="blue")
    plt.fill_between(E, S_E_best_dual - sigma_model_dual, S_E_best_dual + sigma_model_dual,
                 alpha=0.3, color="blue")
    plt.step(centre_best, spectre_best, where='mid' , label='ML BEST Reconstruction', color="cyan")
    plt.fill_between(centre_best, spectre_best - err_y, spectre_best + err_y, color="cyan", alpha=0.3)
    plt.step(centres_Perturbative_ML, spectre_Perturbative_ML, where='mid' , label='Perturbative Algo from ML model', color="orange")
    plt.fill_between(centres_Perturbative_ML, spectre_Perturbative_ML - err_Perturbative_ML, spectre_Perturbative_ML + err_Perturbative_ML, color="orange", alpha=0.5)
    plt.step(centres_Perturbative_Scratch, spectre_Perturbative_Scratch, where='mid' , label='Perturbative Algo from Uniform', color="gray")
    plt.fill_between(centres_Perturbative_Scratch, spectre_Perturbative_Scratch - err_Perturbative_Scratch, spectre_Perturbative_Scratch + err_Perturbative_Scratch, color="gray", alpha=0.5)
    plt.step(centres_Perturbative_dual, spectre_Perturbative_dual, where='mid' , label='Perturbative Algo from 2T Fit', color="green")
    plt.fill_between(centres_Perturbative_dual, spectre_Perturbative_dual - err_Perturbative_dual, spectre_Perturbative_dual + err_Perturbative_dual, color="green", alpha=0.3)
    
    plt.xlabel("Énergie (keV)")
    plt.ylabel("Spectre S(E)")
    plt.yscale("log")  # souvent utile pour un spectre exponentiel
    plt.xscale("log")  # souvent utile pour un spectre exponentiel
    plt.grid(True, which="both", ls="--")
    plt.ylim(1, 1e6*best_A0)
    plt.xlim(10, 1e6)
    plt.legend(fontsize=8)
    plt.title(metadata)
    
    # Chemin complet du fichier (nom au choix)
    filename = os.path.join(path, filename+"_Minuit_Spectre.png")

    # Sauvegarde
    plt.savefig(filename)
    
    plt.show() 
    
    
##############################################################################################    
##############################################################################################    
##############################################################################################    
    
def objective_relative(ip_model, ip_measured, epsilon=1e-1):
    return np.sum(((ip_model - ip_measured) / (0.1*ip_measured + 1e-8)) ** 2)


##############################################################################################    
##############################################################################################    
##############################################################################################    

def inversion_pm_relative(
    psl_measured, S_E, energy_centers, Matrice,
    sigma_low, sigma_high, N,
    epsilon=1e-8, flag_spectrum=True,
    tol=1e-6, patience=2000
):
    """
    tol       = amélioration minimale de l'erreur pour continuer
    patience  = nombre max d'itérations sans amélioration
    """
    n_bins = Matrice.shape[1]

    if flag_spectrum==True:
        log_incident_current = np.log(S_E + epsilon)
    else:
        log_incident_current = np.zeros(n_bins)

    best_log_incident = log_incident_current.copy()
    ip_current = Matrice @ np.exp(log_incident_current)

    best_error = objective_relative(ip_current, psl_measured, epsilon)

    # Passage en log pour les perturbations
    sigma_low, sigma_high = np.log(sigma_low), np.log(sigma_high)

    # Compteur pour l'early stopping
    no_improve_count = 0

    for i in range(N):
        log_perturbations = np.zeros(n_bins)

        start, end = 0, min(45, n_bins)
        if end > start:
            log_perturbations[start:end] = np.random.uniform(
                sigma_low, sigma_high, size=end - start
            )

        candidate_log_incident = log_incident_current + log_perturbations
        candidate_incident = np.exp(candidate_log_incident)

        candidate_smoothed = savgol_filter(candidate_incident, 5, 3)
        candidate_smoothed = np.clip(candidate_smoothed, 0, None)

        ip_model = Matrice @ candidate_incident  # ou candidate_smoothed

        error = objective_relative(ip_model, psl_measured, epsilon)

        if error < best_error - tol:
            best_error = error
            best_log_incident = np.log(candidate_incident + epsilon)
            log_incident_current = best_log_incident
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Si pas d'amélioration depuis trop longtemps, on arrête
        if no_improve_count >= patience:
            #print(f"  → Arrêt anticipé à l’itération {i} (patience atteinte)")
            break

        # Si l'erreur est déjà très petite
        if best_error < tol:
            #print(f"  → Arrêt anticipé à l’itération {i} (tolérance atteinte)")
            break

    return np.exp(best_log_incident), best_error

##############################################################################################    
##############################################################################################    
##############################################################################################    

# --- Plusieurs runs ---
def multiple_runs(psl_measured, S_E, energy_centers, Matrice, sigma_low, sigma_high, N, M, epsilon=1e-8, flag_spectrum=True):
    reconstructions, errors = [], []

    for m in trange(M, desc="Perturbative Algo in progress"):
        recon, err = inversion_pm_relative(
            psl_measured, S_E, energy_centers, Matrice,
            sigma_low, sigma_high, N=N, epsilon=epsilon,
            flag_spectrum=flag_spectrum
        )
        reconstructions.append(recon)
        errors.append(err)

    reconstructions = np.array(reconstructions)
    mean_recon = np.mean(reconstructions, axis=0)
    std_recon = np.std(reconstructions, axis=0)
    median_recon = np.median(reconstructions, axis=0)
    mad = np.median(np.abs(reconstructions - median_recon), axis=0)
    # Convertir MAD en sigma approx.
    robust_std_recon = mad / 0.6745

    #return mean_recon, std_recon, errors    
    return mean_recon, robust_std_recon, errors  



##############################################################################################    
##############################################################################################    
##############################################################################################    

    
def Get_Results_from_Perturbative(centre_best, spectre_best, flag_spectrum):
    # --- Lancement ---
    new_edges, new_centres, S_E = rebin_spectre_personnalise(centre_best, spectre_best, edges_uniforme, binning_scheme)
    
    if flag_spectrum == True:
        mean_incident, std_incident, errors = multiple_runs(spectre_IP, S_E, new_centres, Matrice, sigma_low=0.7, sigma_high=1.3, N=200000, M=50, flag_spectrum=flag_spectrum)
    else :
        mean_incident, std_incident, errors = multiple_runs(spectre_IP, S_E, new_centres, Matrice, sigma_low=0.5, sigma_high=5, N=200000, M=50, flag_spectrum=flag_spectrum)

    
    # --- PSL reconstruit moyen ---
    psl_model_mean = Matrice @ mean_incident
    # Calcul des variances du modèle
    sigma2_ip_model = (Matrice**2) @ (std_incident**2)
    
    # Et les écarts-types
    sigma_ip_model = np.sqrt(sigma2_ip_model)

    return new_centres, mean_incident, std_incident, psl_model_mean, sigma_ip_model


