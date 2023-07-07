import pandas as pd
from sklearn.preprocessing import StandardScaler
from traitement_commentaires import selection_chemin, data_fichier, verification_dossier, \
    traitement_entier_commentaires
from traitement_adresses import traitement_entier_adresse
from joblib import dump, load


def separer_en_deux_dataframes(dataframe):
    """
    :param dataframe: un dataframe avec des commentaires et des adresses
    :return: deux dataframe, un avec les colonnes de commentaires et un avec les colonnes d'adresse
    """
    df_adresses = pd.DataFrame([[]])
    df_commentaires = pd.DataFrame([[]])
    colonnes_adresses = ["adresse_client", "adresse_pb"]
    colonnes_commentaires = ["COMMENTAIRE_DE_SIG", "COMMENTAIRE", "BLOC_NOTE", "COMMLITIGEPIDI",
                             "COMMENTCONTACT"]
    for colonne in dataframe.columns:
        if colonne in colonnes_adresses:
            print("adresse :" + colonne)
            df_adresses = pd.concat([df_adresses, dataframe[colonne]], axis=1)
        elif colonne in colonnes_commentaires:
            print("commentaires :" + colonne)
            df_commentaires = pd.concat([df_commentaires, dataframe[colonne]], axis=1)
    print(df_commentaires.shape, df_adresses.shape)
    return df_commentaires, df_adresses


def mutualisation(chemin_fichier=None, chemin_dossier=None):
    """
    Traite les colonnes de commentaires et d'adresse du fichier sélectionné ou passé en argument et enregistre les
    résultats dans le dossier passé en argument ou sélectionné.
    """
    dossier = ""
    essai = 0
    limite = 3
    dataframe = pd.DataFrame([])
    selection = chemin_fichier is None or chemin_dossier is None
    selection_reussie_f = False
    selection_reussie_d = False

    # Si besoin, on sélectionne un fichier
    if selection:
        while (not selection_reussie_f or not selection_reussie_d) and (essai <= limite):
            essai += 1
            fichier, dossier = selection_chemin()
            selection_reussie_f, dataframe = data_fichier(fichier)
            # Peut-on sauvegarder dans ce dossier ?
            selection_reussie_d = verification_dossier(dossier)

    else:
        fichier = chemin_fichier
        dossier = chemin_dossier
        type_fichier = fichier[-3:]
        try:
            if type_fichier == "csv":
                dataframe = pd.read_csv(fichier)
            else:
                dataframe = pd.read_excel(fichier)
            selection_reussie_f = True
            selection_reussie_d = True
        except ValueError:
            print("Fichier du mauvais type ou illisible.")

    if selection_reussie_f and selection_reussie_d:
        try:
            # On garde la colonne avec le numéro de désignation
            df_designation = dataframe["iar_ndfictif"]

            # On sépare en 2 le dataframe
            df_commentaires, df_adresses = separer_en_deux_dataframes(dataframe)

            # On sauvegarde les 2 dataframes
            path_commentaires = dossier + "/df_commentaires.xlsx"
            path_adresses = dossier + "/df_adresses.xlsx"
            df_commentaires.to_excel(path_commentaires, index=False)
            df_adresses.to_excel(path_adresses, index=False)

            # On fait le traitement des commentaires et des adresses séparément
            df_traite_commentaires = traitement_entier_commentaires(chemin_fichier=path_commentaires,
                                                                    chemin_dossier=dossier)
            df_traite_adresses = traitement_entier_adresse(chemin_fichier=path_adresses, chemin_dossier=dossier)

            # On applique une normalisation aux colonnes :
            df_norm_commentaires = df_traite_commentaires
            df_norm_adresses = df_traite_adresses

            # On concatène les résultats
            dataframe_concat = pd.concat([df_designation, df_norm_commentaires, df_norm_adresses], axis=1)

            # On enregistre les résultats
            dataframe_concat.to_pickle(dossier + "/df_prepare.pkl")
            dataframe_concat.to_excel(dossier + "/df_prepare.xlsx", index=False)

            return dataframe_concat, dossier
        except ValueError:
            print("Soucis dans la préparation des données.")
        except KeyError:
            print("Soucis dans la préparation des données.")
    else:
        print("Problème de sélection du dossier d'adresse.")
        return None, None


def df_to_listedf(df, nbr_lignes):
    """
    :param df: un dataframe
    :param nbr_lignes: le nombre de lignes maximal par sous-df.
    :@return: une liste de dataframes correspondant au df passé en argument séparé en plusieurs morceaux.
    """
    borne_inf = 0
    borne_sup = nbr_lignes - 1
    limite = len(df)
    dfs = []
    ajout = True
    while ajout:
        dfs.append(df.loc[borne_inf:borne_sup])

        # Si la dernière ligne ajoutée est la dernière ligne du dataframe
        if borne_sup == limite - 1:
            ajout = False

        # S'il y a nbr_lignes lignes ou moins à ajouter
        elif borne_sup + nbr_lignes > limite:
            borne_sup = borne_sup + 1
            dfs.append(df.loc[borne_sup:limite])
            ajout = False

        # Sinon
        else:
            borne_inf = borne_sup + 1
            borne_sup = borne_sup + nbr_lignes

    return dfs


def creation_df_intermediaires(path_eti, path_dossier):
    liste_path = []
    df_eti31 = pd.read_excel(path_eti)
    dfs_eti31 = df_to_listedf(df_eti31, 30)

    compteur = 1
    for df in dfs_eti31:
        path_df = path_dossier + "/df_intermediaire" + str(compteur) + ".xlsx"
        df.to_excel(path_df, index=False)
        liste_path.append(path_df)
        compteur += 1
    return liste_path, dfs_eti31


def mutualisation_eti31():
    path_eti31 = "C:/Users/SPML0410/Documents/MachineLearning2/projet/label_adresse_commentaire_eti31_bis.xlsx"
    path_dossier = "C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/eti31/fichiers_calcul"
    path_dossier_final = "C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/eti31/"

    liste_path, dfs = creation_df_intermediaires(path_eti31, path_dossier)

    df_final = pd.DataFrame([[]])
    for i in range(len(liste_path)):
        df_traite = mutualisation(chemin_fichier=liste_path[i], chemin_dossier=path_dossier)
        df_traite_label = pd.concat([df_traite, dfs[i]["Reussite"]], axis=1)
        df_final = pd.concat([df_final, df_traite_label], axis=0)

    df_final.to_excel(path_dossier_final + "/df_final.xlsx", index=False)

    return None


def creation_scaler():
    df_reference = pd.read_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/eti31/df_final.xlsx")
    df_reference = df_reference.drop(columns=["Reussite", "iar_ndfictif"], axis=1)
    scaler = StandardScaler()
    scaler.fit(df_reference)
    dump(scaler, 'C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/scaler2.joblib')


creation_scaler()


def normalisation_for_all(df, exceptions):
    """
    :param df: un dataframe panda
    :param exceptions: Liste des colonnes de df qui n'ont pas à être normalisées
    :return: Le même df avec ses données normalisées.
    """
    # On enlève les colonnes sans données numériques.
    df_a_normaliser = df.drop(exceptions, axis=1)

    # On applique la normalisation standard
    scaler = load('C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/scaler2.joblib')
    donnees_normalisees = scaler.transform(df_a_normaliser)
    df_donnees_normalisees = pd.DataFrame(donnees_normalisees, columns=df_a_normaliser.columns)

    # On remplace dans le nouveau dataframe les anciennes colonnes par les colonnes avec les donnes normalisées
    for colonne in exceptions:
        df_donnees_normalisees[colonne] = df[colonne]

    return df_donnees_normalisees


def normalisation_eti31():
    df = pd.read_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/eti31/df_final.xlsx")
    df_normalise = normalisation_for_all(df, ["Reussite", "iar_ndfictif"])
    df_normalise.to_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/eti31_traite.xlsx", index=False)


# normalisation_eti31()

def complet(chemin_fichier=None, chemin_dossier=None):
    df, dossier = mutualisation(chemin_fichier=chemin_fichier, chemin_dossier=chemin_dossier)
    df_normalise = normalisation_for_all(df, ["iar_ndfictif"])
    df_normalise.to_excel(dossier + "/df_traite.xlsx", index=False)


complet()
