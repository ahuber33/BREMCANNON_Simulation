import os
import shutil

def remplacer_ligne_et_renommer(dossiers, ligne_a_remplacer, nouvelle_ligne, ancien_motif, nouveau_motif, extensions=None):
    """
    Remplace une ligne spécifique dans tous les fichiers d'un ou plusieurs dossiers et les renomme,
    y compris dans les sous-dossiers spécifiés.
    Ne modifie que les fichiers ayant les extensions spécifiées.

    :param dossiers: Liste de dossiers contenant les fichiers à modifier.
    :param ligne_a_remplacer: Texte de la ligne à remplacer.
    :param nouvelle_ligne: Texte de remplacement.
    :param ancien_motif: Motif à remplacer dans le nom des fichiers.
    :param nouveau_motif: Nouveau motif pour renommer les fichiers.
    :param extensions: Liste des extensions des fichiers à traiter (par exemple, ['cpp', 'txt']).
    """
    # Si aucune extension n'est fournie, tout traiter
    if extensions is None:
        extensions = []

    # Récupérer le répertoire du script (chemin absolu)
    chemin_script = os.path.dirname(os.path.realpath(__file__))
    print(f"Emplacement du script: {chemin_script}")

    # Ajouter le dossier du script lui-même à la liste des dossiers à traiter
    dossiers.append('')

    for dossier_rel in dossiers:
        # Construire le chemin absolu du dossier relatif par rapport à l'emplacement du script
        dossier_absolu = os.path.join(chemin_script, dossier_rel)

        # Vérifier si le dossier existe
        if not os.path.exists(dossier_absolu):
            print(f"Le dossier {dossier_absolu} n'existe pas.")
            continue

        for fichier in os.listdir(dossier_absolu):
            chemin_fichier = os.path.join(dossier_absolu, fichier)

            # Vérifier si c'est un fichier et s'il a une extension valide
            if os.path.isfile(chemin_fichier) and (not extensions or any(fichier.endswith(f".{ext}") for ext in extensions)):
                # Créer une copie de sauvegarde avant modification
                #shutil.copy(chemin_fichier, chemin_fichier + ".bak")

                # Lire et modifier le fichier
                try:
                    with open(chemin_fichier, 'r', encoding='utf-8') as f:
                        lignes = f.readlines()
                except UnicodeDecodeError:
                    # Si une erreur de décodage se produit, essayer avec un encodage différent
                    try:
                        with open(chemin_fichier, 'r', encoding='ISO-8859-1') as f:
                            lignes = f.readlines()
                    except UnicodeDecodeError:
                        print(f"Erreur de décodage pour le fichier : {chemin_fichier}")
                        continue  # Passer au fichier suivant

                # Modifier chaque ligne qui correspond exactement à "ligne_a_remplacer"
                for i in range(len(lignes)):
                    if ligne_a_remplacer in lignes[i]:
                        lignes[i] = lignes[i].replace(ligne_a_remplacer, nouvelle_ligne)  # Remplacer le texte

                # Écrire les modifications dans le fichier
                with open(chemin_fichier, 'w', encoding='utf-8') as f:
                    f.writelines(lignes)

                # Renommer le fichier s'il contient l'ancien motif
                if ancien_motif in fichier:
                    nouveau_nom = fichier.replace(ancien_motif, nouveau_motif)
                    os.rename(chemin_fichier, os.path.join(dossier_absolu, nouveau_nom))
                    print(f"Fichier renommé : {fichier} -> {nouveau_nom}")

    print("Modification terminée.")

# Exemple d'utilisation
dossiers_cibles = ["src", "include", "bin"]  # Liste des dossiers relatifs au script
remplacer_ligne_et_renommer(
    dossiers=dossiers_cibles,
    ligne_a_remplacer="CEDRICLULI2023Sim",
    nouvelle_ligne="BREMCANNONSim",
    ancien_motif="CEDRICLULI2023Sim",
    nouveau_motif="BREMCANNONSim",
    extensions=["cpp", "hh", "txt", "cc", "cfg"]
)
