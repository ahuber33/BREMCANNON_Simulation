## Commit #1 [BREMCANNONSim.0.0.1] 
- Previous commits -> See CEDRICLULI2023Sim folder

## Commit #2 [BREMCANNONSim.0.1.0] le 04/03/2025 
- Afin de gagner de l'espace sur les fichiers de sorties, les données ne sont plus enregistrées sous la forme d'un Tree mais directement incorporé dans un TH1D avec un binning fin permettant un possible rebinning lors de l'analyse.

- Cette modification a été nécessaire pour la mise en place d'un grand nombre de simulations visant à faire du Machine Learning.

- Le dossier d'analyse précédent a été gardé mais se trouve désormais sous la notation suivante "backup_v0_Analyse"

- La nouvelle version avec ML se situe dans le dossier "ANALYSE_Config_15IP" car bien sur le Machine Learning effectué correspond à une version particulière de la configuration de CEDRIC utilisé. Si changement il doit y avoir, il faut refaire des données d'entrainements et de tests pour le ML. Un README spécifique à l'utilisation de l'analyse a été édité ainsi qu'un README concernant la mise en place du ML (TO DO)

- Avant, déjà dans un soucis d'espace et de taille des fichiers, seuls les évènements ou il y avait eu une interaction avec CEDRIC était enregistré dans le fichier. Vu que ce n'est plus le cas, les spectres enregistrés sous forme d'histos correspondent à la stat totale simulée.

## Commit #3 [BREMCANNONSim.0.1.1] le 24/04/2025
- Modification de la géométrie de CEDRIC pour correspondre à la configuration exacte de l'expérience GSI 2025
- Modification du code d'analyse afin de correspondre aux spectres détectés à GSI (plus hautes énergies, possible betattron à basses énergies)
- Ajout de 2 autres modèles pour comparaisons et mise en place des incertitudes liées aux modèles.

## Commit #4 [BREMCANNONSim.0.1.2] le 07/07/2025
- MAJ Analyse CEDRIC 15 IP