from geopy.distance import geodesic as GD
import requests
import socket
import os
import pandas as pd
import warnings
import preparation
import tkinter as tk
from tkinter.filedialog import askopenfilenames, askdirectory
socket.getaddrinfo('localhost',8888)
warnings.filterwarnings('ignore')

print("Fin des import")
# Proxy
os.environ['HTTP_PROXY'] = f'http://{"localhost"}:{"8888"}'
os.environ['HTTPS_PROXY'] = f'http://{"localhost"}:{"8888"}'

# Fonctions pour les conversions entre adresse en GPS
# et le calcul de la distance.


def adresse_latitude_longitude(adresse):
    """
    :param adresse: Une adresse la plus détaillée possible.
    :return: La latitude et la longitude de l'adresse.
    """
    api_key = "192349ff78a82f7905ff72c0850de2cd"
    country = "FR"
    url = f"http://api.positionstack.com/v1/forward?access_key={api_key}&query={adresse}&country={country}"
    response = requests.get(url)
    data = response.json()

    if data["data"]:
        # Extraire les informations de géocodage
        try:
            latitude = data["data"][0]["latitude"]
            longitude = data["data"][0]["longitude"]
        except:
            print("erreur (data liste) pour :" + adresse)
            latitude = None
            longitude = None
        return latitude, longitude
    else:
        print("erreur (adresse non trouvée) pour :" + adresse)
        return None, None


def distance_lat_long(a_lat, a_long, b_lat, b_long):
    """
    :param a_lat: Latitude du point A
    :param a_long: Longitude du point A
    :param b_lat: Latitude du point B
    :param b_long: Longitude du point B
    :return: la distance en mètres entre le point A et le point B
    """
    a = (a_lat, a_long)
    b = (b_lat, b_long)

    distance = round(GD(a, b).m, 2)
    print("La distance est de : " + str(distance) + " mètres.")
    return distance


def distance_2_adresses(adresse_a, adresse_b):
    """
    :param adresse_a: L'adresse du point A
    :param adresse_b: L'adresse du point B
    :return: La distance entre le point A et le point B
    """
    a_lat, a_long = adresse_latitude_longitude(adresse_a)
    b_lat, b_long = adresse_latitude_longitude(adresse_b)
    if a_lat is not None and a_long is not None and b_lat is not None and b_long is not None:
        return distance_lat_long(a_lat, a_long, b_lat, b_long)
    else:
        print("Erreur")
        return None


def colonne_adresse(df):
    """
    :param df: Un dataframe panda avec des colonnes de bout d'adresse
    "DI_ADR_NUMV", "DI_ADR_LIBVOIE", "DI_ADR_LIBCOMM".
    :return: la colonne avec l'adresse complète.
    """
    new_col = []
    for i in range(0,len(df)):
        new_col.append(str(df.loc[i,"DI_ADR_NUMV"]) + " " + str(df.loc[i,"DI_ADR_LIBVOIE"]) + " " + str(df.loc[i,"DI_ADR_LIBCOMM"]))
    return new_col


def traitement_adresse(chemin_fichier=None, chemin_dossier=None):
    """
    :param chemin_fichier: (optionnel) pour définir le chemin au fichier à traiter
    :param chemin_dossier: (optionnel) pour définir le chemin au dossier où sauvegarder le pickle
    :return: le chemin du dossier où est enregistrer le pickle du dataframe avec juste une colonne pour l'adresse
    """
    dossier = ""
    essai = 0
    limite = 3
    dataframe = pd.DataFrame([[]])
    selection = chemin_fichier is None or chemin_dossier is None
    selection_reussie_f = False
    selection_reussie_d = False

    if selection:
        while (not selection_reussie_f or not selection_reussie_d) and (essai <= limite):
            essai += 1
            fichier, dossier = preparation.selection_chemin()
            selection_reussie_f, dataframe = preparation.data_fichier(fichier)
            # Peut-on sauvegarder dans ce dossier ?
            selection_reussie_d = preparation.verification_dossier(dossier)

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
            if not("CODE RELEVE" in dataframe.columns):
                vraies_colonnes = ["No DESIGNATION", "DI_ADR_NUMV", "DI_ADR_LIBVOIE", "DI_ADR_LIBCOMM"]
                dataframe = preparation.bonnes_colonnes(dataframe, vraies_colonnes)
            else:
                vraies_colonnes = ["No DESIGNATION", "DI_ADR_NUMV", "DI_ADR_LIBVOIE", "DI_ADR_LIBCOMM", "CODE RELEVE"]
                dataframe = preparation.bonnes_colonnes(dataframe, vraies_colonnes)
                dataframe["Reussite"] = preparation.colonne_label(dataframe, "CODE RELEVE", preparation.label_echec)
                dataframe = dataframe.drop(columns="CODE RELEVE",axis=0)

            dataframe = dataframe.dropna(
                subset=['No DESIGNATION', "DI_ADR_NUMV", "DI_ADR_LIBVOIE", "DI_ADR_LIBCOMM"]).reset_index(
                drop=True)
            dataframe["ADRESSE"]=colonne_adresse(dataframe)
            dataframe = dataframe.drop(columns=["DI_ADR_NUMV","DI_ADR_LIBVOIE","DI_ADR_LIBCOMM"])
            dataframe.to_pickle(dossier + "/df_colonne_adresse.pkl")
            return dossier
        except ValueError:
            print("Les données ont un problème.")

    else:
        print("La preparation des données n'a pas fonctionné.")
        return ""


def liste_adresses_listes_gps(liste_adresses):
    """
    :param liste_adresses: une liste de string correspondant à des adresses
    :return: la liste de latitude et la liste de longitude de cette adresse
    """
    liste_latitude = []
    liste_longitude = []
    longueur = len(liste_adresses)
    i = 0
    for adresse in liste_adresses:
        if i%50 == 0:
            print(str(i) + " sur " + str(longueur))
        i = i+1
        lat_adresse,long_adresse = adresse_latitude_longitude(adresse + "Bretagne France")
        liste_latitude.append(lat_adresse)
        liste_longitude.append(long_adresse)

    return liste_latitude, liste_longitude


def df_to_listedf(df, nbr_lignes):
    """
    :para
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

        # Si il y a nbr_lignes lignes ou moins à ajouter
        elif borne_sup + nbr_lignes > limite:
            borne_sup = borne_sup + 1
            dfs.append(df.loc[borne_sup:limite])
            ajout = False

        # Sinon
        else:
            borne_inf = borne_sup + 1
            borne_sup = borne_sup + nbr_lignes

    return dfs


def traitement_gps_pickle(chemin_fichier=None, chemin_dossier=None, debut=0):
    """
    Le processus de calucl des coordonnées GPS est très long. Pour éviter les mauvaises surprises, on sépare les calculs
    de coordonnées GPS des adresses en plusieurs parties. Le paramètre debut permet de préciser si on s'occupe de tout
    le dataset où si on reprend plus loin. On indiquera donc dans cette variable le numéro du dernier fichier pkl créé.
    :param chemin_fichier: Adresse mémoire d'un fichier excel ou csv avec l'adresse en plusieurs bouts.
    :param chemin_dossier: Adresse mémoire du dossier où sont enregistrés les pickles.
    :param debut: Entier indiquant si on fait la picklisation sur tout le dataframe où si on reprend plus loin.
    :return: La liste des adresses mémoires où sont enregistré les dataframe avec la latitude et la longitude.
    """
    liste_adresses = []
    df = pd.DataFrame([[]])
    selection = False
    try :
        chemin_dossier = traitement_adresse(chemin_fichier=chemin_fichier, chemin_dossier=chemin_dossier)
        df = pd.read_pickle(chemin_dossier + "/df_colonne_adresse.pkl")
        selection = True
    except ValueError:
        print("Problème pour l'obtention dataframe d'adresse.")

    if selection:
        try:
            nbr_lignes = len(df) % 5
            liste_df = df_to_listedf(df, nbr_lignes)
            taille_liste_df = len(liste_df)
            for i in range(debut,taille_liste_df):
                print("On fait le " + str(i) + "ème sur " + str(taille_liste_df))
                df_i = liste_df[i]
                lat, long =  liste_adresses_listes_gps(df_i["ADRESSE"])

                if (lat, long) == (None, None):
                    # Au cas où bug
                    lat, long = liste_adresses_listes_gps(df_i["ADRESSE"])

                df_i["Latitude"],df_i["Longitude"] = lat, long
                name = chemin_dossier + "/gps/df" + str(i) + "_lat_long.pkl"
                liste_adresses.append(name)
                df_i.to_pickle(name)
        except ValueError:
            print("Erreur lors de la conversion en données GPS")

    return liste_adresses


def pickle_to_df_gps(chemin_pickle=[]):
    new_df = pd.DataFrame({'No DESIGNATION': [], 'Reussite': [], "ADRESSE": [], "Latitude": [], "Longitude": []})
    for chemin in chemin_pickle:
        df_i = pd.read_pickle(chemin)
        new_df = pd.concat([new_df, df_i], ignore_index=True)
    return new_df


root = tk.Tk()
root.withdraw()


def traitement_complet(chemin_fichier=None, chemin_dossier=None):
    """
    Enregistre des pickle et si besoin ouvre de fenêtres pour demander où enregistrer.
    :param chemin_dossier: (optionnel) indique où enregistrer les pickles.
    :param chemin_fichier: (optionnel) indique quel fichier prendre pour les adresses.
    :return: un dataframe avec les coordonnées GPS des adresses des clients.
    """
    df = pickle_to_df_gps(traitement_gps_pickle(chemin_fichier=chemin_fichier, chemin_dossier=chemin_dossier, debut=0))
    if chemin_dossier is None:
        while True:
            try:
                chemin_dossier = askdirectory(title="Choisir le répertoire de sauvegarde", mustexist=False, parent=root)
                break
            except ValueError:
                print("Il faut choisir un dossier de sauvegarde !")
    df.to_pickle(chemin_dossier + '/df_coordonnees_gps.pkl')
    return df


def dossier_to_df(chemin_dossier=None):
    """
    Si besoin ouvre une fenêtre pour choisir plusieurs fichiers contenant les adresses de latitude et de longitude.
    Enregistre un dataframe concaténation de ceux sélectionné.
    :param chemin_dossier: (optionnel) Dossier dans lequel on va récupérer tous les fichiers pickles présents.
    :return: Le dataframe correspondant à la concaténation de ceux qui ont été sélectionnés.
    """
    dfs = askopenfilenames(title="Choisissez le fichier à importer", parent=root)
    df = pickle_to_df_gps(dfs)
    if chemin_dossier is None:
        while True:
            try:
                chemin_dossier = askdirectory(title="Choisir le répertoire de sauvegarde", mustexist=False, parent=root)
                break
            except ValueError:
                print("Il faut choisir un dossier de sauvegarde !")
    df.to_pickle(chemin_dossier + '/df_coordonnees_gps.pkl')
    return df

# Attention si programmation d'autres trucs ensuite :
    # Certaines lignes n'auront peut-être pas de coordonnées GPS donc il faudra mettre une erreur à cette ligne et
    # peut-être afficher un petit message d'erreur.


dossier_to_df()

string = "'PBO-85-039-499-3016 // 0 RUE DU ROCHER 85140 CHAUCHE - HauteurPBO : 0.0000 - LocalisationPBO: 22   " \
         "RUE DE LA MARE 85140 CHAUCHE - TypePBO: CHAMBRE - TypeMaterielPBO: _NA_ - CodeAccesImmeuble: _NA_ - " \
         "ContactsImmeuble: _NA_ - InfoObtentionCle: _NA_ - RaccordementLong: Non - ContactsSyndic: _NA_ - " \
         "Pmaccessible: _NA_ - CodeAccesSousSol: _NA_ - CodeLocalPM: _NA_ - ConditionsSyndic: _NA_"
# Faire un split avec les // et les -
# puis récupérer avec contains "LocalisationPBO" et "TypePBO"
# il me semble que RaccordementLong n'est pas toujours rempli

