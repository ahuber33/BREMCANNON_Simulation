# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:39:37 2025

@author: huber
"""
import pickle
import pandas as pd
import numpy as np
import os
# Définir la variable d'environnement pour désactiver oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import Include


##############################################################################
########################### BEGIN Define for user ############################
##############################################################################

root_models = 'D:/Simulations/BREMCANNON_Simulation/ANALYSE_Config_15IP/'
root_save = root_models + '/Save/'       #Important to change this PATH for your own use !!!!
filename = "Shot_test"

##############################################################################
############################# END Define for user ############################
##############################################################################



##############################################################################
################# BEGIN Define ML Configuration Files ########################
##############################################################################

def Load_Configuration_Files():
    # Charger les scalers
    with open(root_models + 'scalers/' + 'scaler_X_31scaled.pkl', 'rb') as f:
        scaler_X_31 = pickle.load(f)    
    with open(root_models + 'scalers/' + 'scaler_X_35scaled.pkl', 'rb') as f:
        scaler_X_35 = pickle.load(f)        
    
    with open(root_models + 'scalers/' + 'scaler_Y_31scaled.pkl', 'rb') as f:
        scaler_Y_31 = pickle.load(f)        
    with open(root_models + 'scalers/' + 'scaler_Y_35scaled.pkl', 'rb') as f:
        scaler_Y_35 = pickle.load(f)            
    
    
    with open(root_models + 'gpy/' + 'GP_Model_31scaled.pkl', 'rb') as file:
        gp_model_31 = pickle.load(file)   
    with open(root_models + 'gpy/' + 'GP_Model_35scaled.pkl', 'rb') as file:
        gp_model_35 = pickle.load(file)       
    
    
    with open(root_models + 'rf/' + 'RF_Model_31scaled.pkl', 'rb') as file:
        rf_model_31 = pickle.load(file)    
    with open(root_models + 'rf/' + 'RF_Model_35scaled.pkl', 'rb') as file:
        rf_model_35 = pickle.load(file)        
    
    #nn_model = load_model(r"D:\PYTHON\Data_ML_extended\nn\NN_Model_35scaled.keras", custom_objects={'MCDropout': MCDropout})
    
    # Charger le DataFrame depuis un fichier pickle
    Matrice = pd.read_pickle(root_models + "matrice_CEDRIC.pkl")
    Matrice=Matrice.T

    return (scaler_X_31, scaler_X_35, scaler_Y_31, scaler_Y_35,
        gp_model_31, gp_model_35,
        rf_model_31, rf_model_35, Matrice)


##############################################################################
################### END Define ML Configuration Files ########################
##############################################################################


def init_all():
    global centres_uniforme, edges_uniforme, edges_non_uniforme, E
    global edges_Minuit, binning_scheme, centres_Minuit, dummy
    global scaler_X_31, scaler_X_35, scaler_Y_31, scaler_Y_35
    global gp_model_31, gp_model_35
    global rf_model_31, rf_model_35, Matrice
    global centre_best, spectre_best, err_x, erry
    global spectre_IP, sigma_spectre_IP
    global m_single, m_dual
    global centres_Perturbative_ML, spectre_Perturbative_ML, err_Perturbative_ML
    global centres_Perturbative_Scratch, spectre_Perturbative_Scratch, err_Perturbative_Scratch
    global centres_Perturbative_dual, spectre_Perturbative_dual, err_Perturbative_dual
    global PSL_ML_P, sigma_PSL_ML_P
    global PSL_P, sigma_PSL_P
    global PSL_dual_P, sigma_PSL_dual_P

    # Charger modèles / scalers
    (scaler_X_31, scaler_X_35, scaler_Y_31, scaler_Y_35,
     gp_model_31, gp_model_35,
     rf_model_31, rf_model_35, Matrice) = Load_Configuration_Files()

    # Générer spectres
    E = np.linspace(0, 500000, 5000)  # adapte Emin/Emax à ton cas
    binning_scheme = [
        (0, 1000, 100),
        (1000, 10000, 1000),
        (10000, 100000, 5000),
        (100000, 500000, 50000)
        ]
    edges_Minuit = np.arange(0, 500000+100, 100)
    centres_Minuit = (edges_Minuit[:-1] + edges_Minuit[1:]) / 2
    centres_uniforme, dummy, edges_uniforme = Include.generer_spectre_uniforme(bin_size=100, total_flux=1)
    edges_non_uniforme, centres_non_uniforme, new_dummy = Include.rebin_spectre_personnalise(centres_uniforme, dummy, edges_uniforme, binning_scheme)
  


    Include.centres_uniforme = centres_uniforme
    Include.edges_uniforme = edges_uniforme
    Include.edges_non_uniforme = edges_non_uniforme
    Include.edges_Minuit = edges_Minuit
    Include.binning_scheme = binning_scheme
    Include.centres_Minuit = centres_Minuit
    Include.dummy = dummy
    Include.E = E
    
    Include.scaler_X_31 = scaler_X_31
    Include.scaler_X_35 = scaler_X_35
    Include.scaler_Y_31 = scaler_Y_31
    Include.scaler_Y_35 = scaler_Y_35
    Include.gp_model_31 = gp_model_31
    Include.gp_model_35 = gp_model_35
    Include.rf_model_31 = rf_model_31
    Include.rf_model_35 = rf_model_35
    Include.Matrice = Matrice
    
    print("✅ Initialisation terminée : modèles + scalers + spectres prêts")



##############################################################################
################### BEGIN RECONSTRUCTION SPECTRE ALGO ########################
##############################################################################

init_all()

### CHARGEMENT DONNES DU TIR ###
spectre_IP, sigma_spectre_IP, metadata = Include.Load_ShotFile(root_models, filename + ".txt")
Include.spectre_IP = spectre_IP
Include.sigma_spectre_IP = sigma_spectre_IP

### APPLICATION ML MODEL -> OBTENTION "MEILLEUR" SPECTRE ###
centre_best, spectre_best, err_x, err_y = Include.Estimation_Best_ML(root_models, centres_uniforme)

Include.centre_best = centre_best
Include.spectre_best = spectre_best
Include.err_x = err_x
Include.err_y = err_y


### FIT DES DONNES AVEC EXPO 1T ET 2T AVEC MINUIT ###
m_single, m_dual = Include.Fit_Minuit()
Include.m_single = m_single
Include.m_dual = m_dual

### APPLICATION METHODE PERTURBATIVE SUR LE "MEILLEUR" SPECTRE ML
new_centres, S_ML_P, sigma_S_ML_P, PSL_ML_P, sigma_PSL_ML_P = Include.Get_Results_from_Perturbative(centre_best, spectre_best, flag_spectrum=True)
# Rebin inverse : reconvertir le spectre non uniforme vers binning uniforme
centres_Perturbative_ML, spectre_Perturbative_ML, err_Perturbative_ML = Include.rebin_to_uniform_with_uncertainty(new_centres, S_ML_P, sigma_S_ML_P,
                                      edges_non_uniforme, edges_uniforme)

Include.PSL_ML_P = PSL_ML_P
Include.sigma_PSL_ML_P = sigma_PSL_ML_P
Include.centres_Perturbative_ML = centres_Perturbative_ML
Include.spectre_Perturbative_ML = spectre_Perturbative_ML
Include.err_Perturbative_ML = err_Perturbative_ML


### APPLICATION METHODE PERTURBATIVE SUR SPECTRE UNIFORME (NO A PRIORI)
new_centres, S_P, sigma_S_P, PSL_P, sigma_PSL_P = Include.Get_Results_from_Perturbative(centre_best, spectre_best, flag_spectrum=False)
# Rebin inverse : reconvertir le spectre non uniforme vers binning uniforme
centres_Perturbative_Scratch, spectre_Perturbative_Scratch, err_Perturbative_Scratch = Include.rebin_to_uniform_with_uncertainty(new_centres, S_P, sigma_S_P,
                                      edges_non_uniforme, edges_uniforme)

Include.PSL_P = PSL_P
Include.sigma_PSL_P = sigma_PSL_P
Include.centres_Perturbative_Scratch = centres_Perturbative_Scratch
Include.spectre_Perturbative_Scratch = spectre_Perturbative_Scratch
Include.err_Perturbative_Scratch = err_Perturbative_Scratch


### APPLICATION METHODE PERTURBATIVE SUR SPECTRE FIT 2T
S_E_best_dual = Include.dual_spectrum(E, m_dual.values["A0"], m_dual.values["E01"], m_dual.values["A1"], m_dual.values["E02"])  # shape (500,)
new_centres, S_dual_P, sigma_S_dual_P, PSL_dual_P, sigma_PSL_dual_P = Include.Get_Results_from_Perturbative(centre_best, S_E_best_dual, flag_spectrum=True)
# Rebin inverse : reconvertir le spectre non uniforme vers binning uniforme
centres_Perturbative_dual, spectre_Perturbative_dual, err_Perturbative_dual = Include.rebin_to_uniform_with_uncertainty(new_centres, S_dual_P, sigma_S_dual_P,
                                      edges_non_uniforme, edges_uniforme)

Include.PSL_dual_P = PSL_dual_P
Include.sigma_PSL_dual_P = sigma_PSL_dual_P
Include.centres_Perturbative_dual = centres_Perturbative_dual
Include.spectre_Perturbative_dual = spectre_Perturbative_dual
Include.err_Perturbative_dual = err_Perturbative_dual


### PLOT ###
Include.Plot_Fit_Minuit_IP(root_save, filename, metadata)
Include.Plot_Fit_Minuit_Spectre(root_save, filename, metadata)