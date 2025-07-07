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

############################################################
############### Define for user ############################
############################################################
#root_models = 'D:/PYTHON/ANALYSE_Config_15IP/'  #Important to change this PATH for your own use !!!!
#root_save = root_models + '/Save/'       #Important to change this PATH for your own use !!!!

##### Define for user #####
root_models = 'D:/Simulations/BREMCANNON_Simulation/ANALYSE_Config_15IP/'
root_save = root_models + '/Save/'       #Important to change this PATH for your own use !!!!
filename = "Shot_test"


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
    
    
    #with open(root_models + 'Meta/Meta_GP_Model_31scaled.pkl', 'rb') as f:
    #    meta_model_31 = pickle.load(f)
    #with open(root_models + 'Meta/Meta_GP_Model_35scaled.pkl', 'rb') as f:
    #    meta_model_35 = pickle.load(f)    
    
    meta_model_31=0
    meta_model_35=0
    
    
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
    Matrice = pd.read_pickle(r"D:\PYTHON\Data_ML_extended\matrice_CEDRIC.pkl")

    return (scaler_X_31, scaler_X_35, scaler_Y_31, scaler_Y_35,
        meta_model_31, meta_model_35,
        gp_model_31, gp_model_35,
        rf_model_31, rf_model_35, Matrice)




(scaler_X_31, scaler_X_35, scaler_Y_31, scaler_Y_35,
 meta_model_31, meta_model_35,
 gp_model_31, gp_model_35,
 rf_model_31, rf_model_35, Matrice) = Load_Configuration_Files()


spectre_IP, sigma_spectre_IP, metadata = Include.Load_ShotFile(root_models, filename + ".txt")

centre_best, spectre_best, err_x, err_y = Include.Estimation_Best_ML(
                                            root_save, filename, metadata,
                                            scaler_X_31, scaler_X_35, scaler_Y_31, scaler_Y_35,
                                            meta_model_31, meta_model_35,
                                            gp_model_31, gp_model_35,
                                            rf_model_31, rf_model_35,
                                            spectre_IP)

m_single, m_dual = Include.Fit_Minuit(spectre_best, spectre_IP, sigma_spectre_IP)

Include.Plot_Fit_Minuit_IP(root_save, filename, metadata, m_single, m_dual, spectre_IP, sigma_spectre_IP)
Include.Plot_Fit_Minuit_Spectre(root_save, filename, metadata, m_single, m_dual, centre_best, spectre_best, err_x, err_y)