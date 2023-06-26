#Importations

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn import tree
import matplotlib.pyplot as plt
from textblob import TextBlob
from statistics import mean
from math import ceil
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import graphviz
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re

path_eti31 = "../CSV/ETI31_commentaires.csv"
eti31 = pd.read_csv(path_eti31,low_memory=False)
label_echec = ["ANC", "ANN", "ETU", "MAJ", "ORT", "PAD", "PBC", "REO", "RMC", "RMF", "RRC"]


# -----------------------------------------------------------------------------------------------------------------------------
#Dans cette partie on enlève les lignes qui ne sont pas utilisable et on labelise les données
def colonne_label(df, colonne,label):
    """
    :return: la colonne correspondante a la labelisation a partir des codes de releves
    """
    new_colonne = []
    for i in df[colonne]:
        if i in label:
            new_colonne.append(False)
        else :
            new_colonne.append(True)
    return new_colonne
def nettoyageEtLabelisation(df):
    """
    :param df: Le data frame à nettoyer et labéliser
    :return: df nettoyé avec les labels correspondants
    """
    #Nettoyage
    df = df.dropna(subset=['CODE RELEVE',])
    df = df.dropna(subset=["COMMENTAIRE DE SIG","COMMENTAIRE","BLOC NOTE","COMMLITIGEPIDI"],how='all')
    df = df.reset_index(drop=True)

    #Labelisation
    df["Reussite"] = colonne_label(df,"CODE RELEVE",label_echec)
    df = df.drop(columns="CODE RELEVE")
    return df

dataframe = nettoyageEtLabelisation(eti31)
dataframe.to_pickle("../PICKLE/dataframe_labelise.pkl")

# -----------------------------------------------------------------------------------------------------------------------------
#Dans cette partie on nettoye les commentaires et plutôt que d'avoir plusieurs colonnes de commentaires on a une seule colonne avec une liste de commentaires.
labels = ["rrc","dms","pad","tvc","etu","rmc","rmf","ann","ort","anc","pbc","reo"]
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
def anti_acronymes_grammar(liste,dico):
    """
    :param liste: une liste de string (commentaires de techniciens)
    :return: la liste de string dans lesquels ont a remplacé les mots selon le dictonnaire passé en argument
    """

    liste_retour = []
    for e in liste:
        for key, value in dico.items():
            e = e.replace(key, value)
        liste_retour.append(e)

    return liste_retour
def lower_case(liste):
    return [e.lower() for e in liste]
def sans_numero_ponctuation(liste):
    liste_rendu = []
    pattern = r'\d+'
    for text in liste :
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub(pattern,'',text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        liste_rendu.append(text)
    return liste_rendu
def liste_propre(liste):
    """
    :param liste: une liste de commentaires
    :return: la liste de commentaires nettoyes
    """
    #On met tout en minuscule
    liste = lower_case(liste)
    #On enleve les codes de releve
    liste = enleve_releve_liste(liste)
    #On enleve les signes et on remplaces les accronymes et erreurs principales de frappe
    dico_mots = {"=": "", ".": "", "_": " ", "-": " ", "/": " ", "$": "", "#": "", "teste": "test",
                 "nimporte": "n'importe", "cable": "câble", "c ble": "câble", "tranch e": "tranchée",
                 "tranchee": "tranchée", "tranche": "tranchée", "tiquetage": "étiquetage", "malfa": "malfaçon",
                 "pb": "point de branchement", "mevo": "message vocal", "pto": "point de terminaison optique",
                 "pto_non": "point de terminaison optique non", "ligible": "éligible", "racc": "raccordement",
                 "fa ade": "facade", "propi taire": "propriétaire", "detect e": "détectée"}
    liste = anti_acronymes_grammar(liste,dico_mots)
    #On enleve les nombres et la ponctuation
    liste = sans_numero_ponctuation(liste)
    return liste

def transformation_colonnes_commentaires(df, colonnes, separateur):
    """
    :param df: un dataframe contenant au moins une colonne de commentaires
    :param colonnes: liste de string
    :param separateur: un caractère qui permet de séparer le string pour chaque élément de la liste
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
                commentaires_i = commentaires_i + liste_propre(texte.split(separateur))

        colonne_commentaires.append(commentaires_i)
    return colonne_commentaires
def drop_pas_comm(df):
    for index in df.index:
        if df.loc[index,"COMMENTAIRES"] == []:
            df = df.drop(df.index[index])
    return df
def nettoyageDesCommentaires(df):
    df["COMMENTAIRES"] = transformation_colonnes_commentaires(df, ["COMMENTAIRE DE SIG", "COMMENTAIRE", "BLOC NOTE","COMMLITIGEPIDI"], '\\')
    df = df.drop(columns=["COMMENTAIRE DE SIG", "COMMENTAIRE", "BLOC NOTE", "COMMLITIGEPIDI"])
    df = drop_pas_comm(df)
    return df
dataframe = nettoyageDesCommentaires(dataframe)
dataframe.to_pickle("../PICKLE/df_liste_commentaires.pkl")

# -----------------------------------------------------------------------------------------------------------------------------
#Dans cette partie on créer 2 nouvelles colonnes dans le dataframe qui indique le score d'echec et le score de reusssite à partir du dataframe de référence d'ETI31.
def liste_to_string(liste):
    final = ""
    for commentaire in liste :
        final = final + commentaire
    return final

def colonne_liste_to_colonne_string(colonne):
    final_colonne = []
    for i in colonne :
        final_colonne.append(liste_to_string(i))
    return final_colonne

#On recupere le dataframe des scores
dataframe_scores = pd.read_pickle("../pickle/df_proportions_mots.pkl")
def score_type(commentaires,reussite,dataframe_scores):
    """

    :param commentaires: un string
    :param reussite: 0 si on veut le score de reussite, 1 pour celui d'echec
    :param dataframe_scores: le dataframe contenant les score de chaque mot
    :return: la colonne contenant le score de reussite ou d'echec pour chaque ligne
    """
    score = 0
    for mot in dataframe_scores.columns :
            nb_occurence = commentaires.count(mot)
            score = score + nb_occurence*dataframe_scores.loc[reussite,mot]
    return score
def colonne_scores(df,colonne):
    score_reussite = []
    score_echec = []
    for row in df[colonne]:
        score_reussite.append(score_type(row,0,dataframe_scores))
        score_echec.append(score_type(row,1,dataframe_scores))
    return(score_reussite,score_echec)

def scoreEchecEtReussite(df):
    df["ensemble_commentaires"] = colonne_liste_to_colonne_string(df["COMMENTAIRES"])
    df["score_reussite"], df["score_echec"] = colonne_scores(df, "ensemble_commentaires")
    df.drop(columns="ensemble_commentaires")
    return df

dataframe = scoreEchecEtReussite(dataframe)
dataframe.to_pickle("../PICKLE/df_score.pkl")

# -----------------------------------------------------------------------------------------------------------------------------
#Dans cette partie on fait des statistiques pour le rapport de stage et des statistiques pour le machine learning
def proportions(df):
    count_reussite = df["Reussite"].value_counts(normalize=True)*100
    # Création d'un histogramme des pourcentages
    count_reussite.plot(kind='bar')

    # Ajout d'un titre et d'étiquettes d'axe
    plt.title('Histogramme des pourcentages par valeur')
    plt.xlabel('Valeur')
    plt.ylabel('Pourcentage')

    # Affichage du graphique
    plt.show()
    print(count_reussite)

#proportions()
def nombres_de_commentaires(dataframe):
    ensemble = dataframe["COMMENTAIRES"]
    nombre_commentaires = []
    somme = 0
    for commentaires in ensemble:
        i = len(commentaires)
        somme = somme + i
        nombre_commentaires.append(i)
    moyenne = somme/len(ensemble)
    print("Le nombre moyen de commentaires est de " + str(moyenne)+".")
    return nombre_commentaires


def longueur_moyenne_commentaires(dataframe):
    ensemble = dataframe["COMMENTAIRES"]
    longueur_moyenne_commentaire = []
    somme_ensemble = 0

    for commentaires in ensemble:
        somme = 0
        for commentaire in commentaires:
            somme = somme + len(commentaire)

        moyenne_commentaires = somme / len(commentaires)
        somme_ensemble = somme_ensemble + moyenne_commentaires

        longueur_moyenne_commentaire.append(moyenne_commentaires)
    moyenne_ensemble = somme_ensemble / len(ensemble)
    print("La longueur moyenne des commentaires est de " + str(moyenne_ensemble) + ".")
    return longueur_moyenne_commentaire

def histogramme(df,colonne,bins,titre):
    plt.hist(df[colonne],bins=bins)
    plt.title(titre)
    plt.show()
def stats(df):
    proportions(df)
    df_echec = df[df["Reussite"] == False]
    df_reussite = df[df["Reussite"] == True]
    df_echec["nombre_commentaires"] = nombres_de_commentaires(df_echec)
    df_reussite["nombre_commentaires"] = nombres_de_commentaires(df_reussite)
    df["nombre_commentaires"] = nombres_de_commentaires(df)
    df_echec["Longueur moyenne commentaire"] = longueur_moyenne_commentaires(df_echec)
    df_reussite["Longueur moyenne commentaire"] = longueur_moyenne_commentaires(df_reussite)
    df["Longueur moyenne commentaire"] = longueur_moyenne_commentaires(df)

    histogramme(df, "nombre_commentaires", 10,
                "Répartition du nombre de commentaires pour une intervention pour l'ensemble")
    histogramme(df, "Longueur moyenne commentaire", 10,
                "Répartition de la longueur moyenne des commentaires d'une intervention pour l'ensemble")
    histogramme(df_echec, "nombre_commentaires", 10,
                "Répartition du nombre de commentaires pour une intervention pour les échecs")
    histogramme(df_echec, "Longueur moyenne commentaire", 10,
                "Répartition de la longueur moyenne des commentaires d'une intervention pour les échecs")
    histogramme(df_reussite, "nombre_commentaires", 10,
                "Répartition du nombre de commentaires pour une intervention pour les réussites")
    histogramme(df_reussite, "Longueur moyenne commentaire", 10,
                "Répartition de la longueur moyenne des commentaires d'une intervention pour les réussites")

    #A votre plaisir d'afficher ce que vous voulez
stats(dataframe)

#--------------------------------------------------------------------------------------------------------------------------------
#Dans cette partie on s'occupe de normaliser les données