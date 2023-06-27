# Importations
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import warnings
import re

warnings.filterwarnings('ignore')

label_echec = ["ANC", "ANN", "ETU", "MAJ", "ORT", "PAD", "PBC", "REO", "RMC", "RMF", "RRC"]


# -----------------------------------------------------------------------------------------------------------------------------
# Dans cette partie, on enlève les lignes qui ne sont pas utilisable et on labellise les données
def colonne_label(df, colonne, label):
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
    df = df.drop(df[df["CODE RELEVE"] == ""].index, axis=0)
    df = df.dropna(subset=["COMMENTAIRE DE SIG", "COMMENTAIRE", "BLOC NOTE", "COMMLITIGEPIDI"], how='all')
    df = df.reset_index(drop=True)

    # Labelisation
    df["Reussite"] = colonne_label(df, "CODE RELEVE", label_echec)
    df = df.drop(columns="CODE RELEVE")
    return df


def nettoyage_donnes_en_cours(df):
    """
    :param df: Le dataframe à nettoyer et labelliser
    :return: df nettoyé avec les labels correspondants
    """
    df = df.dropna(subset=["COMMENTAIRE DE SIG", "COMMENTAIRE", "BLOC NOTE", "COMMLITIGEPIDI"], how='all')
    df = df.reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------------------------------------------------------
# Dans cette partie, on nettoie les commentaires et plutôt que d'avoir plusieurs colonnes de commentaires,
# on a une seule colonne avec une liste de commentaires.
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
    return [e.lower() for e in liste]


def sans_numero_ponctuation(liste):
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
    for index in df.index:
        if not df.loc[index, "COMMENTAIRES"]:
            df = df.drop(df.index[index])
    return df


def nettoyage_des_commentaires(df):
    df["COMMENTAIRES"] = transformation_colonnes_commentaires(df, ["COMMENTAIRE DE SIG", "COMMENTAIRE", "BLOC NOTE",
                                                                   "COMMLITIGEPIDI"], '\\')
    df = df.drop(columns=["COMMENTAIRE DE SIG", "COMMENTAIRE", "BLOC NOTE", "COMMLITIGEPIDI"])
    df = drop_pas_comm(df)
    return df


# -----------------------------------------------------------------------------------------------------------------------------
# Dans cette partie, on crée 2 nouvelles colonnes dans le dataframe qui indique
# le score d'echec et le score de reusssite à partir du dataframe de référence d'ETI31.


def liste_to_string(liste):
    final = ""
    for commentaire in liste:
        final = final + commentaire
    return final


def colonne_liste_to_colonne_string(colonne):
    final_colonne = []
    for i in colonne:
        final_colonne.append(liste_to_string(i))
    return final_colonne


# On recupere le dataframe des scores
dataframe_scores = pd.read_pickle("pickle/df_proportions_mots.pkl")


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
    score_reussite = []
    score_echec = []
    for row in df[colonne]:
        score_reussite.append(score_type(row, 0, dataframe_scores))
        score_echec.append(score_type(row, 1, dataframe_scores))
    return score_reussite, score_echec


def score_echec_et_reussite(df):
    df["ensemble_commentaires"] = colonne_liste_to_colonne_string(df["COMMENTAIRES"])
    df["score_reussite"], df["score_echec"] = colonne_scores(df, "ensemble_commentaires")
    df = df.drop(columns="ensemble_commentaires")
    return df


# -----------------------------------------------------------------------------------------------------------------------------
# Dans cette partie on fait des statistiques pour le rapport de stage et des statistiques pour le machine learning
def proportions(df):
    count_reussite = df["Reussite"].value_counts(normalize=True) * 100
    # Création d'un histogramme des pourcentages
    count_reussite.plot(kind='bar')

    # Ajout d'un titre et d'étiquettes d'axe
    plt.title('Histogramme des pourcentages par valeur')
    plt.xlabel('Valeur')
    plt.ylabel('Pourcentage')

    # Affichage du graphique
    plt.show()
    print(count_reussite)


# proportions()
def nombres_de_commentaires(df):
    ensemble = df["COMMENTAIRES"]
    nombre_commentaires = []
    somme = 0
    for commentaires in ensemble:
        i = len(commentaires)
        somme = somme + i
        nombre_commentaires.append(i)
    moyenne = somme / len(ensemble)
    print("Le nombre moyen de commentaires est de " + str(moyenne) + ".")
    return nombre_commentaires


def longueur_moyenne_commentaires(df):
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
    moyenne_ensemble = somme_ensemble / len(ensemble)
    print("La longueur moyenne des commentaires est de " + str(moyenne_ensemble) + ".")
    return longueur_moyenne_commentaire


def histogramme(df, colonne, bins, titre):
    plt.hist(df[colonne], bins=bins)
    plt.title(titre)
    plt.show()


def stats(df):
    df["nombre_commentaires"] = nombres_de_commentaires(df)
    df["Longueur moyenne commentaire"] = longueur_moyenne_commentaires(df)

    """"
    proportions(df)
    
    Si besoin d'afficher les histogrammes :
    
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
    """
    return df

    # A votre plaisir d'afficher ce que vous voulez


# --------------------------------------------------------------------------------------------------------------------------------
# Dans cette partie, on s'occupe de normaliser les données
def normalisation_for_all(df, exceptions):
    # On enlève les colonnes sans donnees numeriques
    df_a_normaliser = df.drop(exceptions, axis=1)

    # On applique la normalisation standard
    scaler = StandardScaler()
    donnees_normalisees = scaler.fit_transform(df_a_normaliser)
    df_donnees_normalisees = pd.DataFrame(donnees_normalisees, columns=df_a_normaliser.columns)

    # On remplace dans le nouveau dataframe les anciennes colonnes par les colonnes avec les donnes normalisées
    for colonne in exceptions:
        df_donnees_normalisees[colonne] = df[colonne]

    return df_donnees_normalisees


# --------------------------------------------------------------------------------------------------------------------------------
# Polarisation et subjectivite des mots
def polarite_liste_commentaires(liste):
    """
    :param liste: une liste de string
    :return: une liste contenant la polarite de chaque string
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
    moyenne = somme_ensemble / len(liste)
    print(moyenne)
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
    moyenne = somme_ensemble / len(liste)
    print(moyenne)
    return liste_subjectivite


def polarite_et_subjectivite(df):
    df["polarite"] = polarite_liste_commentaires(df["COMMENTAIRES"])
    df["subjectivite"] = subjectivite_liste_commentaires(df["COMMENTAIRES"])
    return df


def selection_chemin():
    # Demander un fichier
    root = tk.Tk()
    root.withdraw()
    fichier = askopenfilename(title="Choisissez le fichier à importer")
    print(fichier)
    # Demander un endroit où mettre les fichiers pkl
    dossier = askdirectory(title="Choisir le répertoire de sauvegarde", mustexist=False)
    print(dossier)
    return fichier, dossier


def data_fichier(fichier):
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


def bonnes_colonnes(df, col):
    vraies_colonnes = col
    colonnes_a_enlever = []
    for colonne in df.columns:
        if colonne in vraies_colonnes:
            vraies_colonnes.remove(colonne)
        else:
            colonnes_a_enlever.append(colonne)

    taille = len(df)
    df = df.drop(columns=colonnes_a_enlever,axis=0)
    for colonne in vraies_colonnes:
        if colonne == "No DESIGNATION":
            df[colonne] = [1 for x in range(0, taille)]
        else :
            df[colonne] = ["" for x in range(0, taille)]
    return df


def traitement_entier(selection=True, chemin_fichier=None, chemin_dossier=None, code_releve=False):
    dossier = ""
    essai = 0
    limite = 3
    dataframe = pd.DataFrame([])
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
        except ValueError:
            print("Fichier du mauvais type ou illisible.")

    if selection_reussie_f and selection_reussie_d:
        try:
            if not code_releve:
                vraies_colonnes = ["No DESIGNATION", "COMMENTAIRE DE SIG", "COMMENTAIRE", "BLOC NOTE", "COMMLITIGEPIDI"]
                dataframe = bonnes_colonnes(dataframe, vraies_colonnes)
                dataframe = nettoyage_donnes_en_cours(dataframe)
            else:
                vraies_colonnes = ["No DESIGNATION", "COMMENTAIRE DE SIG",	"CODE RELEVE", "COMMENTAIRE", "BLOC NOTE", "COMMLITIGEPIDI"]
                dataframe = nettoyage_et_labelisation(dataframe)

            dataframe.to_pickle(dossier + "/dataframe_labelise.pkl")
            dataframe = nettoyage_des_commentaires(dataframe)
            dataframe.to_pickle(dossier + "/df_liste_commentaires.pkl")
            dataframe = score_echec_et_reussite(dataframe)
            dataframe.to_pickle(dossier + "/df_score.pkl")
            dataframe = stats(dataframe)

            if not code_releve:
                dataframe = normalisation_for_all(dataframe, ["No DESIGNATION", "COMMENTAIRES"]).dropna(axis=0)
            else:
                dataframe = normalisation_for_all(dataframe, ["No DESIGNATION", "Reussite", "COMMENTAIRES"]).dropna(axis=0)
            dataframe = polarite_et_subjectivite(dataframe)
            dataframe = dataframe.drop(columns=["COMMENTAIRES"], axis=0)
            dataframe.to_pickle(dossier + "/df_stat_commentaires.pkl")
            return dossier
        except ValueError:
            print("Les données ont un problème.")

    else:
        print("La preparation des données n'a pas fonctionnée.")
        return ""


traitement_entier(selection=True, code_releve=False)
#code_releve permet de savoir si on prépare des données avec une labelisation ou non.
