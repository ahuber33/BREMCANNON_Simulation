# BREMCANNON ANALYSE with CEDRIC 15 IPs CONFIGURATION (Mylar entrance). [huber@lp2ib.in2p3.fr, huberarnaud@gmail.com]

## INSTRUCTIONS TO USE THE ANALYSYS CODE
- If you already cloned all the BREMCANNON_Simulation files, you have already access to the ANALYSE_Config_15IP codes.

- If you don't want to do some simulations but only use the Analysis code on Windows for example with Spyder, you need to follow these instructions :
    - Go to : https://github.com/ahuber33/BREMCANNON_Simulation/
    - Clik on the green block "<> Code"
    - Download ZIP
    - You can keep only the ANALYSE_Config_15_IP codes


For the Analysis, IN ADDITION; you need to download some pickle files (too big to be uploaded on GIT).

You will find :
- GP_model_31scaled.pkl
- GP_model_35scaled.pkl
- RF_model_31scaled.pkl
- RF_model_35scaled.pkl

with this link : https://sdrive.cnrs.fr/s/TJfikzYww39AosP  (Link avalaible until 31/12/2025)

You need to put these files on gpy folder (for GP model) and rf folder (for RF model) !!!!      
 

- Concerning the Analysis code, first of all, you need to modify the "root_models" PATH in the "Analyse_CEDRIC.py" file acccording to your installation (example below)
```
############################################################
############### Define for user ############################
############################################################
##### Define for user #####
root_models = 'D:/Simulations/BREMCANNON_Simulation/ANALYSE_Config_15IP/'
root_save = root_models + '/Save/'       #Important to change this PATH for your own use !!!!
filename = "Shot_test"


###########################################################
###########################################################
```  
DO NOT CHANGE THE OTHER PATHS EXCEPT IF YOU CHANGE THE STRUCTURE OF FILES/FOLDERS

- After that, according to your environment, you will be available to use the Analysis code with Python. If you just want to use the analysis code, it is HIGHLY RECOMMANDED to not work on the GEANT4 VM due to possible disk spaces and time reconstruction performances.

- If you choose the GEANT4 Virtual Machine environment, you will need to download some module with these lines :
```
sudo dnf install python3-tkinter -y
sudo pip install scikit-learn
sudo pip install tensorflow
```  
- If you have a problem of disk space, follow this example below :
```
bash
mkdir -p PATH/TO/INSTALL/tmp
mkdir -p PATH/TO/INSTALL/python_packages
export TMPDIR=PATH/TO/INSTALL/tmp
pip install --target=PATH/TO/INSTALL/python_packages tensorflow
export PYTHONPATH=PATH/TO/INSTALL/python_packages:$PYTHONPATH
```

- After that, you can use this command :
```
python3 Analyse_CEDRIC.py
```  

- If you choose another environment like SPYDER, according to your version, maybe you will need also to download the previous modules or others. Use the same method on a terminal (pip install ...). After that, you will be able to run it.

- The code will look at the "Shot_test.txt" file which have the following structure :
    - Metadata
    - PSL/mm² IP1
    - PSL/mm² IP2
    - ...

- An uncertainty of 10% is applied for the Data !!!! 

- From this IP spectrum, the code will try to reconstruct an incident spectrum 
    - from ML reconstruction
        - Gaussian Process trained with ONLY Exponentials (31)
        - Gaussian Process trained on different structures (35)
        - Random Forrest trained with ONLY Exponentials (31)
        - Random Forrest trained on different structures (35)
    - from Minuit Fit with 1 or 2 temperatures
    - from Perturbative Minimization Algo PMA (https://pubs.aip.org/aip/rsi/article/95/2/023301/3262492/Robust-unfolding-of-MeV-x-ray-spectra-from-filter)
        - from ML best spectrum
        - from 2T Fit
        - from Uniform spectrum (Scratch)

- At the end, you will obtain on the SAVE folder :
    - Best configuration determined for the ML
    - Other results from ML
    - Superposition of all ML reconstruction
    - Comparison of IP spectrums (ML, Fit, PMA) with Data with Chi2 values
    - Comparison of Incident spectrum (ML, Fit, PMA) 

- For more informations about the ML process, look at the README file locatedin the ML folder.