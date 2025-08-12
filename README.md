# BREMCANNON for Studies of CEDRIC's ENL Bremmsthralung cannon spectrometer. [huber@lp2ib.in2p3.fr, huberarnaud@gmail.com]

## INSTRUCTIONS TO USE THE SIMULATION
- Download the VMWare [Geant4.11.2.1](https://heberge.lp2ib.in2p3.fr/G4VM/index.html)

```
git clone https://github.com/ahuber33/BREMCANNON_Simulation
```

- Go to build Folder and use this command :
```
cmake -DGeant4_DIR=$G4COMP ../
make -j4
```  
then compile it with make

- The executable BREMCANONSim will be add to your bin folder

- If you want to have a visualization, launch this command : 
```
./BREMCANONSim [name of ROOT file ]"
```  
It will generate 1 particle according to the vis.mac with QT and you will have a ROOT file with the name you gave in response located in the Resultats folder.

- If you want to have statistics without the visualization, use this command :
```
./BREMCANONSim [name of ROOT file] [number of events generated] [name of macro] [MultiThreading ON/OFF] [number of threads]
```  
According to the number of threads used if MT is ON, the simulation will create a ROOT file for each thread and at the end of the simulation, all ROOT files will be merged together with a name correspoding to the name given in [name of ROOT file]. The temporaries ROOT files will be removed after the merge.

Note that it's not necessary to indicate a [number of threads] if the condition on MT is OFF. In opposite, you need to put a value if MT is ON.

Personnaly, I used the vrml.mac but you can create another one. Just to remember that you need to write the name of your macro when you launch the simulation.


- An BREMCANONSim.cfg file is located in bin directory. All the dimensions necessary are in this file to avoid recompilation when you want to change some parameters. If you add some other dimensions, don't forget to add the variables in Geometry.cc.
```
#----------Common variables----------
Radius_cylinder_internal 7 mm
Radius_cylinder_external 25 mm
Radius_IP 6.5 mm
Stack_IP_pos_ini 0 mm
Cylinder_tantale_length  80 mm
Disk_thickness 2.5 mm
Mylar_thickness 13 um
flag_Mylar 1 #0=false & 1 = true
IPa1_z 9 um
IPa2_z 115 um
IPa3_z 190 um
IPa4_z 160 um
Number_IP 15
Thickness_filter {1 4 8 1 1 3 4 2 3 2 2 4 8 12}
Material_filter {Alu Alu Alu Cuivre Cuivre Cuivre Cuivre Etain Etain Plomb Plomb Plomb Plomb Plomb}
Labs 222 um
Coef_PSL 6.95e-4
```

- Possibility to switch between a Mylar entrance (13µm) before the first IP and an aluminium disk (thickness 2.5 mm) with the flag_Mylar parameter.

- Tables (Thickness & Materials) are initiated with a max of 20. If needed to have more than 20 IPs, you need to change that on the code.

- Some materials are already defined in the simulation. If you need a new one, you must declare it in the BREMCANONSimGeometry.cc and precisely on the con construction part of interest. If the material is already in the NIST Database, you can copy the declaration and modifiy the declaration to create a new material. If not, it is advice to declare it in the BREMCANONSimMaterials.cc in order to clarify the code. After that, DO NOT FORGET to add the declaration of your new material in the ConstructMaterialsList() function in BREMCANONSimGeometry.cc file. It is NECESSARY if you want to have the conversion of your material name given in the configuration file and the link with the G4Material associated.

- Concerning the geometry, you can change the number of IPs (and so Filters) directly with the config file. Same for the ticknesses/materials of the filters. If the combination of all IPs & Filters length is higher than the available space (Cylindre_tantale_hauteur), the simulation will drop a FatalException at the beginning of the simulation. In the other case, the available space behind the last IPs will be filled by a lead volume. 

- Based on G4EmStandardPhysics_option3.

- DO NOT HESITATE TO REPORT BUGS OR ANY IDEAS THAT WILL IMPROVE THE SIMULATION !!!!


## BREMCANNON ANALYSE with CEDRIC 15 IPs CONFIGURATION (Mylar entrance). [huber@lp2ib.in2p3.fr, huberarnaud@gmail.com]

## INSTRUCTIONS TO USE THE ANALYSYS CODE
- If you already cloned all the BREMCANNON_Simulation files, you have already access to the ANALYSE_Config_15IP codes.

- If you don't want to do some simulations but only use the Analysis code on Windows for example with Spyder, you need to follow these instructions :
    - Go to : https://github.com/ahuber33/BREMCANNON_Simulation/
    - Clik on the green block "<> Code"
    - Download ZIP
    - You can keep only the ANALYSE_Config_15_IP codes
    - Go to README on this folder

For the Analysis, IN ADDITION; you need to download some pickle files (too big to be uploaded on GIT).

You will find :
- GP_model_31scaled.pkl
- GP_model_35scaled.pkl
- RF_model_31scaled.pkl
- RF_model_35scaled.pkl

with this link : https://sdrive.cnrs.fr/s/TJfikzYww39AosP  (Link avalaible until 31/12/2025)