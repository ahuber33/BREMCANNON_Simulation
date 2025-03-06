# BREMCANNON ANALYSE with CEDRIC 15 IPs CONFIGURATION (Mylar entrance). [huber@lp2ib.in2p3.fr, huberarnaud@gmail.com]

## INSTRUCTIONS TO USE THE ANALYSYS CODE
- If you already cloned all the BREMCANNON_Simulation files, you have already access to the ANALYSE_Config_15IP codes.

- If you don't want to do some simulations but only use the Analysis code on Windows for example with Spyder, you need to follow these instructions :
    - Go to : https://github.com/ahuber33/BREMCANNON_Simulation/
    - Clik on the green block "<> Code"
    - Download ZIP
    - You can keep only the ANALYSE_Config_15_IP codes

- First of all, you need to modify the "root_models" PATH in the "Analyse_CEDRIC.py" file acccording to your installation (example below)
```
############################################################
############### Define for user ############################
############################################################
root_models = 'D:/PYTHON/ANALYSE_Config_15IP/'  #Important to change this PATH for your own use !!!!
root_save = root_models + '/Save/'
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

- A graphical window will pop up in order, for users, to put the informations (Name of file & PSL/mm² for each IP).

- When it's good, click on the button "Proceed" to use the reconstruction made by the Machine Learning process.

- You will obtain a plot corresponding to the estimated gamma spectrum incident on the solid angle of the CEDRIC detector. This plot will be automatically save with the filename given in the graphical interface in the Save folder

- If you are on the GEANT4 VM, you need to close the first plot (PSL/IP) to have the reconstructed spectrum.

- For more informations about the ML process, look at the README file locatedin the ML folder.