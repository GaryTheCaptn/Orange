from geopy.distance import geodesic as GD
import requests
import socket
import os
import pandas as pd
import warnings
from traitement_commentaires import selection_chemin, data_fichier, verification_dossier, \
    bonnes_colonnes

socket.getaddrinfo('localhost', 8888)
warnings.filterwarnings('ignore')

# Proxy
os.environ['HTTP_PROXY'] = f'http://{"localhost"}:{"8888"}'
os.environ['HTTPS_PROXY'] = f'http://{"localhost"}:{"8888"}'


def reponse_api(adresse):
    """

    """
    api_key = "192349ff78a82f7905ff72c0850de2cd"
    country = "FR"
    url = f"http://api.positionstack.com/v1/forward?access_key={api_key}&query={adresse}&country={country}"
    response = requests.get(url)
    data = response.json()
    return data


def adresse_latitude_longitude(adresse):
    """
    :param adresse: Une adresse la plus détaillée possible.
    :return: La latitude et la longitude de l'adresse.
    """
    data = reponse_api(adresse)

    if isinstance(data, dict):
        # Extraire les informations de géocodage
        try:
            latitude = data["data"][0]["latitude"]
            longitude = data["data"][0]["longitude"]
        except ValueError:
            data2 = reponse_api(adresse)
            try:
                latitude = data2["data"][0]["latitude"]
                longitude = data2["data"][0]["longitude"]
            except ValueError:
                print("erreur (data liste) pour :" + adresse)
                latitude = None
                longitude = None
            except TypeError:
                print("erreur (data liste) pour :" + adresse)
                latitude = None
                longitude = None
        except TypeError:
            data2 = reponse_api(adresse)
            try:
                latitude = data2["data"][0]["latitude"]
                longitude = data2["data"][0]["longitude"]
            except ValueError:
                print("erreur (data liste) pour :" + adresse)
                latitude = None
                longitude = None
            except TypeError:
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
        return None


def colonne_distance_2_adresses(adresses_a, adresses_b):
    col = []
    for i in range(len(adresses_a)):
        if i % 5 == 0:
            print(i)
        col.append(distance_2_adresses(adresses_a[i], adresses_b[i]))
    return col


def traitement_entier_adresse(chemin_fichier=None, chemin_dossier=None):
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
        vraies_colonnes = ["adresse_client", "adresse_pb"]
        dataframe = bonnes_colonnes(dataframe, vraies_colonnes)
        dataframe["distance"] = colonne_distance_2_adresses(dataframe["adresse_client"], dataframe["adresse_pb"])
        dataframe = dataframe.drop(columns=["adresse_client", "adresse_pb"])
        dataframe.to_excel(dossier + "/df_distance_adresse.xlsx", index=False)
        return dataframe
    else:
        print("Problème de sélection du dossier d'adresse.")
        return None

# traitement_entier()
