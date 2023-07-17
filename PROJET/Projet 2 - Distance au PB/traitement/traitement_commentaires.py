# Importations
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import warnings
import re

warnings.filterwarnings('ignore')


# Index du code :
# Enlever les lignes vides (et labélisation pour ETI31)
# Nettoyage de commentaires (ponctuation, nombre, fautes etc)
# Transformation des colonnes en une liste de string dans une seule colonne COMMENTAIRES
# Colonnes de score de réussite et d'échec à partir des données d'ETI31
# Statistiques sur le nombre de commentaires, la longueur moyenne des commentaires
# Normalisation des données
# Polarité et subjectivité du texte
# Fonctions pour traiter un fichier (sélection et traitement complet)
# Fonctions pour garder et ajouter les bonnes colonnes
# Fonction qui regroupe tout le traitement
# ----------------------------------------------------------------------------------------------------------------------
# Enlever les lignes qui sont vides
def nettoyage_donnes_en_cours(df):
    """
    :param df: Le dataframe à nettoyer
    :return: df sans les lignes qui n'ont aucun commentaire.
    """
    df = df.dropna(subset=["COMMENTAIRE_DE_SIG", "COMMENTAIRE", "BLOC_NOTE", "COMMLITIGEPIDI", "COMMENTCONTACT"],
                   how='all')
    df = df.reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------------------------------------------------
# Uniquement pour ETI31


label = ["ANC", "ANN", "ETU", "MAJ", "ORT", "PAD", "PBC", "REO", "RMC", "RMF", "RRC"]


def colonne_label(df, colonne):
    """
    :return: la colonne correspondante a la labellisation à partir des codes de relèves
    """
    new_colonne = []
    for i in df[colonne]:
        if i in label:
            new_colonne.append(False)
        else:
            new_colonne.append(True)
    return new_colonne


def nettoyage_et_labelisation(df):
    """
    :param df: Le dataframe à nettoyer et labelliser
    :return: df nettoyé avec les labels correspondants
    """
    # Nettoyage
    df = df.drop(df[df["CODE_RELEVE"] == ""].index, axis=0)
    df = df.dropna(subset=["COMMENTAIRE_DE_SIG", "COMMENTAIRE", "BLOC_NOTE", "COMMLITIGEPIDI", "COMMENTCONTACT"],
                   how='all')
    df = df.reset_index(drop=True)

    # Labelisation
    df["Reussite"] = colonne_label(df, "CODE_RELEVE")
    df = df.drop(columns="CODE_RELEVE")
    return df


# Fin uniquement pour ETI31

# ----------------------------------------------------------------------------------------------------------------------
# Fonctions pour nettoyer les commentaires

labels = ["rrc", "dms", "pad", "tvc", "etu", "rmc", "rmf", "ann", "ort", "anc", "pbc", "reo"]


def enleve_releve_liste(commentaires):
    """
    :param commentaires: liste de string contenant des commentaires de techniciens
    :return: la liste des commentaires sans les codes de releve
    """
    commentaires_propres = []
    for commentaire in commentaires:
        for label in labels:
            commentaire = commentaire.replace(label, '')
        commentaires_propres.append(commentaire)

    return commentaires_propres


def anti_acronymes_grammar(liste, dico):
    """
    :param liste: une liste de string (commentaires de techniciens)
    :param dico: dictionnaire indiquant comment remplacer les strings inutiles
    :return: la liste de string dans lesquels on a remplacé les mots selon le dictionnaire passé en argument
    """

    liste_retour = []
    for e in liste:
        for key, value in dico.items():
            e = e.replace(key, value)
        liste_retour.append(e)

    return liste_retour


def lower_case(liste):
    """
    :param liste: une liste de strings
    :return: la même liste avec les strings en miniscule.
    """
    return [e.lower() for e in liste]


def sans_numero_ponctuation(liste):
    """
    :param liste: une liste de strings
    :return: la même liste sans chiffre et sans ponctuation
    """
    liste_rendu = []
    pattern = r'\d+'
    for text in liste:
        text = re.sub('\[.*?', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub(pattern, '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        liste_rendu.append(text)
    return liste_rendu


def liste_propre(liste):
    """
    :param liste: une liste de commentaires
    :return: la liste de commentaires nettoyes
    """
    # On met tout en minuscule
    liste = lower_case(liste)

    # On enlève les codes de releve
    liste = enleve_releve_liste(liste)

    # On enlève les signes et on remplace les acronyms et erreurs principales de frappe
    dico_mots = {"=": "", ".": "", "_": " ", "-": " ", "/": " ", "$": "", "#": "", "teste": "test",
                 "nimporte": "n'importe", "cable": "câble", "c ble": "câble", "tranch e": "tranchée",
                 "tranchee": "tranchée", "tranche": "tranchée", "tiquetage": "étiquetage", "malfa": "malfaçon",
                 "pb": "point de branchement", "mevo": "message vocal", "pto": "point de terminaison optique",
                 "pto_non": "point de terminaison optique non", "ligible": "éligible", "racc": "raccordement",
                 "fa ade": "facade", "propi taire": "propriétaire", "detect e": "détectée"}
    liste = anti_acronymes_grammar(liste, dico_mots)

    # On enlève les nombres et la ponctuation
    liste = sans_numero_ponctuation(liste)

    return liste


# ----------------------------------------------------------------------------------------------------------------------
# Fonctions pour concaténer les colonnes et les différentes interventions en une liste de commentaires pour chaque
# donnée.


def transformation_colonnes_commentaires(df, colonnes, separator):
    """
    :param df: un dataframe contenant au moins une colonne de commentaires
    :param colonnes: liste de string
    :param separator: un caractère qui permet de séparer le string pour chaque élément de la liste
    :return: une liste de liste de String
    """
    # Création de la liste
    colonne_commentaires = []
    # On va procéder ligne à ligne
    for i in df.index:
        # On crée la liste des commentaires pour la ligne i
        commentaires_i = []

        # Pour chaque colonne on split le commentaire et on l'ajoute à la liste des commentaires de la ligne i.
        for colonne in colonnes:
            # On récupère le commentaire
            texte = df.at[i, colonne]
            # S'il y en a un, on applique le séparateur et on rend ça propre
            if isinstance(texte, str):
                commentaires_i = commentaires_i + liste_propre(texte.split(separator))

        colonne_commentaires.append(commentaires_i)
    return colonne_commentaires


def drop_pas_comm(df):
    """
    :param df: un dataframe pandan contenant une colonne COMMENTAIRES
    :return: Le même data frame sans les lignes n'ayant aucun commentaire.
    """
    for index in df.index:
        if not df.loc[index, "COMMENTAIRES"]:
            df = df.drop(df.index[index])
    return df


def nettoyage_des_commentaires(df):
    """
    :param df: dataframe pandas ayant des colonnes de commentaires
    :return: le dataframe avec tous les textes nettoyé et une seule colonne COMMENTAIRE avec la liste des commentaires
    trouvé à partir des autres colonnes.
    """
    df["COMMENTAIRES"] = transformation_colonnes_commentaires(df, ["COMMENTAIRE_DE_SIG", "COMMENTAIRE", "BLOC_NOTE",
                                                                   "COMMLITIGEPIDI", "COMMENTCONTACT"], '\\')
    df = df.drop(columns=["COMMENTAIRE_DE_SIG", "COMMENTAIRE", "BLOC_NOTE", "COMMLITIGEPIDI", "COMMENTCONTACT"])
    df = drop_pas_comm(df)
    return df


# -----------------------------------------------------------------------------------------------------------------------------
# Dans cette partie, on crée 2 nouvelles colonnes dans le dataframe qui indique le score d'echec et le score de
# reusssite à partir du dataframe de référence d'ETI31.


def liste_to_string(liste):
    """
    :param liste: une liste de string
    :return: un string contenant tous les strings de la liste.
    """
    final = ""
    for commentaire in liste:
        final = final + commentaire
    return final


def colonne_liste_to_colonne_string(colonne):
    """
    :param colonne: une liste de liste de string
    :return: la liste avec un seul string au lieu d'une liste de string.
    """
    final_colonne = []
    for i in colonne:
        final_colonne.append(liste_to_string(i))
    return final_colonne


# On récupère le dataframe des scores.
dataframe_scores = pd.read_pickle(
    "/Projet 2 - Distance au PB/traitement/commentaires/df_score_echec_reussite.pkl")


def score_type(commentaires, reussite, df_scores):
    """
    :param commentaires: un string
    :param reussite: 0 si on veut le score de reussite, 1 pour celui d'echec
    :param df_scores: le dataframe contenant le score de chaque mot
    :return: la colonne contenant le score de reussite ou d'echec pour chaque ligne
    """
    score = 0
    for mot in df_scores.columns:
        nb_occurence = commentaires.count(mot)
        score = score + nb_occurence * df_scores.loc[reussite, mot]
    return score


def colonne_scores(df, colonne):
    """
    :param df: un dataframe panda
    :param colonne: une liste de string.
    :return: une liste avec le score de réussite des string de colonne et une liste avec le score d'échec.
    """
    score_reussite = []
    score_echec = []
    for row in df[colonne]:
        score_reussite.append(score_type(row, 0, dataframe_scores))
        score_echec.append(score_type(row, 1, dataframe_scores))
    return score_reussite, score_echec


def score_echec_et_reussite(df):
    """
    :param df: un dataframe panda
    :return: le même dataframe avec une colonne score échec et une colonne score réussite.
    """
    df["ensemble_commentaires"] = colonne_liste_to_colonne_string(df["COMMENTAIRES"])
    df["score_reussite"], df["score_echec"] = colonne_scores(df, "ensemble_commentaires")
    df = df.drop(columns="ensemble_commentaires")
    return df


# -----------------------------------------------------------------------------------------------------------------------------
# Dans cette partie on fait des statistiques pour le rapport de stage et des statistiques pour le machine learning


def proportions(df):
    """
    :param df: une dataframe avec une colonne réussite
    :return: affiche les proportions de chaque classe.
    """
    count_reussite = df["Reussite"].value_counts(normalize=True) * 100
    # Création d'un histogramme des pourcentages
    count_reussite.plot(kind='bar')

    # Ajout d'un titre et d'étiquettes d'axe
    plt.title('Histogramme des pourcentages par valeur')
    plt.xlabel('Valeur')
    plt.ylabel('Pourcentage')

    # Affichage du graphique
    plt.show()
    # print(count_reussite)


# proportions()
def nombres_de_commentaires(df):
    """
    :param df: une dataframe avec une colonne COMMENTAIRES (liste de liste de string)
    :return: la liste avec le nombre de commentaires pour chaque donnée.
    """
    ensemble = df["COMMENTAIRES"]
    nombre_commentaires = []
    somme = 0
    for commentaires in ensemble:
        i = len(commentaires)
        somme = somme + i
        nombre_commentaires.append(i)
    # moyenne = somme / len(ensemble)
    # print("Le nombre moyen de commentaires est de " + str(moyenne) + ".")
    return nombre_commentaires


def longueur_moyenne_commentaires(df):
    """
    :param df: un dataframe avec une colonne COMMENTAIRES (liste de liste de String)
    :return: la longeur moyenne de chaque commentaire de chaque donnée.
    """
    ensemble = df["COMMENTAIRES"]
    longueur_moyenne_commentaire = []
    somme_ensemble = 0

    for commentaires in ensemble:
        somme = 0
        for commentaire in commentaires:
            somme = somme + len(commentaire)

        moyenne_commentaires = somme / len(commentaires)
        somme_ensemble = somme_ensemble + moyenne_commentaires

        longueur_moyenne_commentaire.append(moyenne_commentaires)
    # moyenne_ensemble = somme_ensemble / len(ensemble)
    # print("La longueur moyenne des commentaires est de " + str(moyenne_ensemble) + ".")
    return longueur_moyenne_commentaire


def histogramme(df, colonne, bins, titre):
    """
    :param df: un dataframe
    :param colonne: une colonne du dataframe qui sert de pivot pour l'histogramme
    :param bins: le nombre de sous découpage
    :param titre: titre de l'histogramme
    Affiche l'histogramme.
    """
    plt.hist(df[colonne], bins=bins)
    plt.title(titre)
    plt.show()


def stats(df, affiche=False):
    """
    :param df: un dataframe panda
    :return: le même dataframe avec les statistiques sur ses commentaires dans 2 nouvelles colonnes "nombre_commentaire"
            et "Longueur moyenne commentaire" et si on le souhaite, on peut afficher les histogrammes.
    """
    df["nombre_commentaires"] = nombres_de_commentaires(df)
    df["longueur_moyenne"] = longueur_moyenne_commentaires(df)

    if affiche:
        proportions(df)

        histogramme(df, "nombre_commentaires", 10,
                    "Répartition du nombre de commentaires pour une intervention pour l'ensemble")
        histogramme(df, "longueur_moyenne", 10,
                    "Répartition de la longueur moyenne des commentaires d'une intervention pour l'ensemble")

    return df

    # A votre plaisir d'afficher ce que vous voulez


# --------------------------------------------------------------------------------------------------------------------------------
# Polarisation et subjectivite des mots
def polarite_liste_commentaires(liste):
    """
    :param liste: une liste de string
    :return: une liste contenant la polarité de chaque string
    """
    liste_polarite = []
    somme_ensemble = 0
    for commentaires in liste:
        somme_polarite = 0
        for commentaire in commentaires:
            somme_polarite = somme_polarite + TextBlob(commentaire).polarity
        moyenne_polarite = somme_polarite / len(commentaires)
        somme_ensemble = somme_ensemble + moyenne_polarite
        liste_polarite.append(moyenne_polarite)
    # moyenne = somme_ensemble / len(liste)
    # print(moyenne)
    return liste_polarite


def subjectivite_liste_commentaires(liste):
    """
    :param liste: une liste de string
    :return: une liste de la subjectivite de chaque string
    """
    liste_subjectivite = []
    somme_ensemble = 0
    for commentaires in liste:
        somme_subjectivite = 0
        for commentaire in commentaires:
            somme_subjectivite = somme_subjectivite + TextBlob(commentaire).subjectivity
        moyenne_subjectivite = somme_subjectivite / len(commentaires)
        somme_ensemble = somme_ensemble + moyenne_subjectivite
        liste_subjectivite.append(moyenne_subjectivite)
    # moyenne = somme_ensemble / len(liste)
    # print(moyenne)
    return liste_subjectivite


def polarite_et_subjectivite(df):
    """
    :param df: un dataframe panda avec une colonne de commentaires
    :return: le même datafrale avec deux nouvelles colonnes polarité et subjectivité des commentaires.
    """
    df["polarite"] = polarite_liste_commentaires(df["COMMENTAIRES"])
    df["subjectivite"] = subjectivite_liste_commentaires(df["COMMENTAIRES"])
    return df


# -------------------------------------------------------------------------------------------------------------------------------
# Dans cette partie, on sélectionne les chemins pour accéder aux données et on vérifie qu'ils sont corrects.

root = tk.Tk()
root.withdraw()


def selection_chemin():
    """
    Affiche une fenêtre pour sélectionner un fichier à traiter et faire les prédictions et une fenêtre où enregistrer
    les fichiers pkl.
    """
    # Demander un fichier
    fichier = askopenfilename(title="Choisissez le fichier à importer", parent=root)
    # Demander un endroit où mettre les fichiers pkl
    dossier = askdirectory(title="Choisir le répertoire de sauvegarde", mustexist=False, parent=root)
    return fichier, dossier


def data_fichier(fichier):
    """
    :param fichier: un chemin de fichier
    :return: un booléen qui indique si le fichier est bien lisible et le dataframe du fichier passé en argument.
    """
    type_fichier = fichier[-3:]
    dataframe = pd.DataFrame([])
    res = False

    if fichier != "":
        try:
            if type_fichier == "csv":
                dataframe = pd.read_csv(fichier)
            else:
                dataframe = pd.read_excel(fichier)
            res = True
        except ValueError:
            res = False
            print("Fichier du mauvais type ou illisible.")

    return res, dataframe


def verification_dossier(dossier):
    """
    :param dossier: Un chemin de dossier.
    :return: Un booléen indiquant si on peut bien écrire dans le dossier.
    """
    res = False
    dataframe = pd.DataFrame([])
    try:
        dataframe.to_pickle(dossier + "/test.pkl")
        res = True
    except ValueError:
        print("Erreur dossier invalide.")
    except PermissionError:
        print("Erreur dossier invalide.")
    return res


# -----------------------------------------------------------------------------------------------------------------------------
# Dans cette partie on formate les colonnes


def bonnes_colonnes(df, col):
    """
    :param df: Un dataframe panda.
    :param col: Une liste de colonnes qui doivent être dans le dataframe.
    :return: Le dataframe avec les bonnes colonnes (ajoutées pour celles qui n'existaient pas et supprimées pour celles
    qui ne sont pas bonnes).
    """
    vraies_colonnes = col
    colonnes_a_enlever = []
    for colonne in df.columns:
        if colonne in vraies_colonnes:
            vraies_colonnes.remove(colonne)
        else:
            colonnes_a_enlever.append(colonne)

    taille = len(df)
    df = df.drop(columns=colonnes_a_enlever, axis=0)
    for colonne in vraies_colonnes:
        if colonne == "No DESIGNATION":
            df[colonne] = [1 in range(0, taille)]
        else:
            df[colonne] = ["" in range(0, taille)]
    return df


# -----------------------------------------------------------------------------------------------------------------------------
# Fonction principale qui compile tous les traitements.

"""
def traitement_ETI31():
    df = pd.read_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/ETI31_PB_adresses_commentaires.xlsx")
    # Colonnes = iar_ndfictif, CODE_RELEVE, COMLITIFEPODO, COMMENTCONTACT,BLOC_NOTZ, COMMENTAIRE, COMMENTAIRE_DE_SID,
    # adrese_client, adresse_pb

    df = df.drop(columns=["iar_ndfictif", "adresse_client", "adresse_pb"])
    # Colonnes = CODE_RELEVE, COMLITIFEPODO, COMMENTCONTACT,BLOC_NOTZ, COMMENTAIRE, COMMENTAIRE_DE_SIG

    df = nettoyage_et_labelisation(df)
    # Colonnes = Reussite, COMLITIFEPODO, COMMENTCONTACT,BLOC_NOTZ, COMMENTAIRE, COMMENTAIRE_DE_SIG

    df = nettoyage_des_commentaires(df)
    # Colonnes = Reussite, COMMENTAIRES

    df = score_echec_et_reussite(df)
    # Colonnes = Reussite, COMMENTAIRES, score_echec, score_reussite

    df = stats(df)
    # Colonnes = Reussite, COMMENTAIRES, score_echec, score_reussite, longueur_moyenne,
    # nombre_commentaires

    df = normalisation_for_all(df, ["Reussite", "COMMENTAIRES"]).dropna(axis=0)
    # Colonnes = Reussite, COMMENTAIRES, score_echec, score_reussite, longueur_moyenne,
    # nombre_commentaires

    df = polarite_et_subjectivite(df)
    # Colonnes = Reussite, COMMENTAIRES, score_echec, score_reussite, longueur_moyenne,
    # nombre_commentaires, polarite, subjectivite

    df = df.drop(columns="COMMENTAIRES", axis=0)
    # Colonnes = Reussite, score_echec, score_reussite, longueur_moyenne,nombre_commentaires,
    # polarite, subjectivite

    df.to_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/df_ETI31_COMMENTAIRES.xlsx",
                index=False)


# traitement_ETI31()
"""


def traitement_entier_commentaires(chemin_fichier=None, chemin_dossier=None):
    """
    :param chemin_fichier: (optionnel, default = None) String du chemin d'accès au fichier de commentaires.
    :param chemin_dossier: (optionnel, default = None) String du chemin d'accès au dossier de sauvegarde des dataframes.
    :return: Le chemin du dossier où sont enregistrés les dataframes.
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
            vraies_colonnes = ["COMMENTAIRE_DE_SIG", "COMMENTAIRE", "BLOC_NOTE", "COMMLITIGEPIDI",
                               "COMMENTCONTACT"]
            dataframe = bonnes_colonnes(dataframe, vraies_colonnes)
            dataframe = nettoyage_donnes_en_cours(dataframe)
            dataframe = nettoyage_des_commentaires(dataframe)
            dataframe = score_echec_et_reussite(dataframe)
            dataframe = stats(dataframe)
            dataframe = dataframe.dropna(axis=0)
            dataframe = polarite_et_subjectivite(dataframe)
            dataframe = dataframe.drop(columns=["COMMENTAIRES"], axis=0)
            dataframe.to_excel(dossier + "/df_stat_commentaires.xlsx", index=False)

            return dataframe
        except ValueError:
            print("Les données ont un problème.")

    else:
        print("La preparation des données n'a pas fonctionné.")
        return None

# traitement_entier()
