from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
import time
import joblib
import socket
import warnings


class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class Abr:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root, reussite = self._insert_recursive(self.root, key)
        return reussite

    def _insert_recursive(self, node, key):
        if node is None:
            return TreeNode(key), True

        if key < node.key:
            node.left, reussite = self._insert_recursive(node.left, key)
        elif key > node.key:
            node.right, reussite = self._insert_recursive(node.right, key)
        else:
            return node, False

        return node, reussite

    def pretty_print(self):
        self._pretty_print_recursive(self.root, prefix="", is_left=True)

    def _pretty_print_recursive(self, root, prefix="", is_left=True):
        if root is not None:
            print(prefix + ("|-- " if is_left else "`-- ") + str(root.key))
            next_prefix = prefix + ("|   " if is_left else "    ")
            self._pretty_print_recursive(root.left, next_prefix, True)
            self._pretty_print_recursive(root.right, next_prefix, False)


def create_abr():
    abr_pm = Abr()
    joblib.dump(abr_pm, "abr.joblib")
    return abr_pm


def load_abr():
    abr_pm = joblib.load("abr.joblib")
    return abr_pm


def maj_abr(abr):
    joblib.dump(abr, "abr.joblib")


# abr_pm = create_abr()
abr_pm = load_abr()

warnings.filterwarnings('ignore')

# Proxy
os.environ['HTTP_PROXY'] = f'http://{"localhost"}:{"8888"}'
os.environ['HTTPS_PROXY'] = f'http://{"localhost"}:{"8888"}'
socket.getaddrinfo('localhost', 8888)
download_directory = "C:/Users/SPML0410/Downloads/rapports_pm/"
fichier_texte = "erreur_nd.txt"


def loading_wait_disparition(driver, mot, timeout):
    while timeout > 0:
        try:
            WebDriverWait(driver, 2).until(
                EC.invisibility_of_element_located((By.XPATH, "//*[contains(text(), '" + mot + "}')]")))

        except:
            continue
        else:
            break
    return timeout >= -2


def loading_wait_apparition(driver, path, mot, timeout):
    while timeout > 0:
        try:
            timeout = timeout - 2
            WebDriverWait(driver, 2).until(
                EC.text_to_be_present_in_element((By.XPATH, path), mot)
            )

        except:
            continue
        else:
            break
    return timeout > -2


def bouton_compte_existant(driver):
    try:
        button_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
                                            '/html/body/div[4]/div[1]/div[2]/div/form[2]/table/tbody/tr/td/div[1]/table/tbody/tr/td[2]/a/span'))
        )
        # Si le bouton est présent, on clique.
        button_element.click()
        return True


    except Exception:
        # Sinon, on passe à la suite.
        print("Pas de compte client.")
        return False


def click_button_with_text(driver, keyword, nd):
    try:
        xpath_expression = f"//*[contains(text(), '{keyword}')]"

        element = driver.find_element(By.XPATH, xpath_expression)

        # Clique sur l'élément pour accéder à la page de destination
        element.click()

        # Vérifier si la page de destination contient "AR" dans le chemin XPath "/html/body/div[4]/div[1]/div[1]/div[2]/h1/a"
        if "AR" in driver.find_element(By.XPATH, "/html/body/div[4]/div[1]/div[1]/div[2]/h1/a").text:
            print(f"Élément contenant le mot clé '{keyword}' a été cliqué.")
        else:
            print(f"La page de destination ne contient pas 'AR' dans le chemin XPath.")
    except Exception as e:
        with open(fichier_texte, 'a') as fichier:
            # Ajout de contenu à la fin du fichier
            fichier.write(nd, ", ")
        print("Défaut pour le ND : ", nd)
        driver.close()
        driver.quit()


def is_keyword_present(driver, xpath, keyword):
    try:
        element = driver.find_element(By.XPATH, xpath)
        return keyword in element.text
    except Exception:
        return False


def recherche(nd):
    print(nd)
    # Mise en place
    service = Service(executable_path="/geckodriver.exe")
    firefox_options = Options()
    firefox_options.binary_location = "C:/Program Files/Mozilla Firefox/firefox.exe"
    driver = webdriver.Firefox(service=service, options=firefox_options)
    try:
        # Connexion sur la page principale
        driver.get("https://ipon-ihm.sso.francetelecom.fr/common/uobject.jsp?object=1000")
        # Clique sur connexion Gassi
        driver.find_element(By.XPATH,
                            "/html/body/div[3]/div[1]/div[2]/div/table/tbody/tr/td/div[1]/form/table/tbody/tr/td/div/div/a").click()
        # Clique sur activer
        driver.find_element(By.XPATH,
                            '//*[@id="spanLinkActiver"]').click()

        # Pause chargement de la page
        loading_wait_apparition(driver, '/html/body/div[4]/div[1]/div[1]/div[2]/h1/a', "Inventaire", 120)
        # Clique sur le menu déroulant "Recherche"
        driver.find_element(By.XPATH,
                            '/html/body/div[1]/div/div/div[2]/ul/li[4]/a/span').click()

        """# Clique sur le bouton pour scroller en haut
        driver.find_element(By.XPATH,
                            '/html/body/div[1]/div/div/div[2]/ul/li[4]/ul/ins[8]/a/span').click()"""

        # Clique sur la recherche par compte client dans le menu déroulant
        driver.find_element(By.XPATH,
                            '/html/body/div[1]/div/div/div[2]/ul/li[4]/ul/li[1]/a/span').click()

        # Clique sur le menu déroulant de IAR
        driver.find_element(By.XPATH,
                            '/html/body/div[4]/div[1]/div[2]/div/form[1]/div[1]/table/tbody/tr[2]/td/table/tbody/tr[3]/td/div[1]/table/tbody/tr/td[2]/table/tbody/tr/td[1]/div/div/select').click()
        # Clique sur "contient" dans le menu déroulant
        driver.find_element(By.XPATH,
                            '/html/body/div[4]/div[1]/div[2]/div/form[1]/div[1]/table/tbody/tr[2]/td/table/tbody/tr[3]/td/div[1]/table/tbody/tr/td[2]/table/tbody/tr/td[1]/div/div/select/option[2]').click()
        # Rentre le ND dans la case
        driver.find_element(By.XPATH,
                            '//*[@id="_v7070967307112129805"]').send_keys(nd)
        # Clique sur rechercher
        driver.find_element(By.XPATH, '//*[@id="regularSearchButton"]').click()

        # Attendre la fin de la recherche (disparition du  mot loading)
        loading_wait_disparition(driver, "Loading", 40)

        # On regarde si un résultat a été trouvé
        bouton_clique = bouton_compte_existant(driver)
        if bouton_clique:
            # On vérifie que la page est chargée
            loading_wait_apparition(driver, '//*[@id="t5092962624013011852_0_title"]', 'AT', 20)

            # On clique sur le AT
            driver.find_element(By.XPATH,
                                '/html/body/div[4]/div[1]/div[2]/div/form/table/tbody/tr/td/div[1]/table/tbody/tr/td[2]/a/span').click()

            # On vérifie que la page est chargée
            loading_wait_apparition(driver, '/html/body/div[4]/div[1]/div[1]/div[2]/h1/a', 'AT', 60)

            # On clique sur la route PTF
            driver.find_element(By.XPATH,
                                '/html/body/div[4]/div[1]/div[1]/div[3]/ul/li[4]/a').click()

            # On vérifie que la page est chargée
            loading_wait_apparition(driver,
                                    '/html/body/div[4]/div[1]/div[2]/div/div/form/table/tbody/tr/td/table/tbody[1]/tr/td/span',
                                    'Route PTF', 20)

            time.sleep(8)
            # On clique sur le bouton de l'armoire :
            element_ar = driver.find_element(By.XPATH,
                                             '/html/body/div[4]/div[1]/div[2]/div/form/table/tbody/tr/td/div[1]/table/tbody/tr[11]/td[11]/a')
            if element_ar is not None:
                element_ar.click()

            # On vérifie qu'on est bien sur une armoire :
            text = driver.find_element(By.XPATH, '/html/body/div[4]/div[1]/div[1]/div[2]/h1/a').text
            bool_ar = "AR" in text or "IMB" in text

            if not bool_ar:
                # Si on est pas tombé sur une armoire, on revient en arrière et on clique autre part.
                driver.back()
                time.sleep(8)
                driver.find_element(By.XPATH,
                                    '/html/body/div[4]/div[1]/div[2]/div/form/table/tbody/tr/td/div[1]/table/tbody/tr[13]/td[11]/a').click()
                text = driver.find_element(By.XPATH, '/html/body/div[4]/div[1]/div[1]/div[2]/h1/a').text

                bool_ar = "AR" in text or "IMB" in text

            if bool_ar:
                # On clique sur le PM
                driver.find_element(By.XPATH,
                                    '/html/body/div[4]/div[1]/div[2]/div/form/table/tbody/tr/td/div[1]/table/tbody/tr/td[2]/a/span').click()

                time.sleep(2)

                # On vérifie que la page est chargée
                bool_pm = "Slots" in driver.find_element(By.XPATH, '//*[@id="s_title"]').text

                # On lit le nom du PM et on regarde si on l'a déjà dans l'arbre.
                nom_pm = driver.find_element(By.XPATH,
                                             '/html/body/div[4]/div[1]/div[1]/div[2]/h1/a').text
                insert = abr_pm.insert(nom_pm)
                if insert:
                    if bool_pm:
                        # On clique sur le menu déroulant Operations
                        driver.find_element(By.XPATH,
                                            '/html/body/div[4]/div[1]/div[1]/div[2]/div[1]/ul/li/a/span').click()

                        # On clique sur le bouton de génération du rapport
                        driver.find_element(By.XPATH,
                                            '/html/body/div[4]/div[1]/div[1]/div[2]/div[1]/ul/li/ul/li[14]/a/span').click()

                        time.sleep(1)
                        bool_diag = is_keyword_present(driver, '/html/body/div[3]/div[1]/div[1]/div[2]/h1/span', "Diag")

                        attempt = 0
                        while bool_diag and attempt < 2:
                            time.sleep(30)
                            driver.refresh()
                            bool_diag = is_keyword_present(driver, '/html/body/div[3]/div[1]/div[1]/div[2]/h1/span',
                                                           "Diag")
                            attempt += 1

                        # On vérifie que le rapport a bien été généré.
                        time.sleep(3)
                        bool_gen_rapport = "Rapport enrichi" in driver.find_element(By.XPATH,
                                                                                    '/html/body/div[3]/div[1]/div[1]/div[2]/h1/a').text

                        if bool_gen_rapport:
                            # On clique sur le menu déroulant Operations
                            driver.find_element(By.XPATH,
                                                '/html/body/div[3]/div[1]/div[1]/div[2]/div[1]/ul/li/a/span').click()

                            # On clique sur le bouton de sauvegarde du rapport en Excel
                            driver.find_element(By.XPATH,
                                                '/html/body/div[3]/div[1]/div[1]/div[2]/div[1]/ul/li/ul/li[3]/a/span').click()

                            # Attendre que le fichier soit téléchargé (timeout de 40 secondes)
                            time.sleep(30)
                            driver.close()
                            driver.quit()

                        else:
                            # Ouverture du fichier en mode ajout
                            with open(fichier_texte, 'a') as fichier:
                                # Ajout de contenu à la fin du fichier
                                text = str(nd) + ", "
                                fichier.write(text)
                            print("Erreur: Génération du rapport pour le ND : ", nd)

                            driver.close()
                            driver.quit()
                    else:
                        driver.close()
                        driver.quit()
                else:
                    print(nd, "Rapport déjà téléchargé")
                    driver.close()
                    driver.quit()
            else:
                print(nd, "AR pas trouvé")
                with open(fichier_texte, 'a') as fichier:
                    # Ajout de contenu à la fin du fichier
                    text = str(nd) + ", "
                    fichier.write(text)
                driver.close()
                driver.quit()


        else:
            driver.quit()

    except Exception as e:
        print(e)
        with open(fichier_texte, 'a') as fichier:
            # Ajout de contenu à la fin du fichier
            text = str(nd) + ", "
            fichier.write(text)
        print("Défaut pour le ND : ", nd)
        driver.close()
        driver.quit()


def recherche_liste(liste):
    liste_int = [int(element) for element in liste]
    with ThreadPoolExecutor() as executor:
        executor.map(recherche, liste_int)


def liste_to_listeliste(liste, nbr_elements):
    borne_inf = 0
    borne_sup = nbr_elements - 1
    limite = len(liste)
    ll = []
    ajout = True
    while ajout:
        temp = []
        for i in range(borne_inf, borne_sup + 1):
            temp.append(liste[i])
        ll.append(temp)
        # Si la dernière ligne ajoutée est la dernière ligne du dataframe
        if borne_sup == limite - 1:
            ajout = False

        # S'il y a nbr_lignes lignes ou moins à ajouter
        elif borne_sup + nbr_elements >= limite:
            temp = []
            for i in range(borne_sup + 1, limite):
                temp.append(liste[i])
            ll.append(temp)
            ajout = False

        # Sinon
        else:
            borne_inf = borne_sup + 1
            borne_sup = borne_sup + nbr_elements

    return ll


def sauvegarde_rapport_pm():
    df = pd.read_excel(r"C:\Users\SPML0410\Documents\MachineLearning2\projet\automate_pm\liste_nd.xlsx")
    ll = liste_to_listeliste(df["No DESIGNATION"], 5)
    for i in range(509, len(ll)):
        print(i, "/", len(ll) - 1)
        recherche_liste(ll[i])
        maj_abr(abr_pm)


# sauvegarde_rapport_pm()


def merge_rapports_pm():
    dossier = "C:/Users/SPML0410/Downloads/rapports_pm/"
    temp_dossier = "C:/Users/SPML0410/Downloads/rapports_pm_temp/"
    i = 1
    for path_fichier in os.listdir(dossier):
        if i % 10 == 0:
            print(i, "/668")
        chemin_fichier = os.path.join(dossier, path_fichier)
        df_temp = pd.read_excel(chemin_fichier, header=None,
                                skiprows=4)  # Lire le fichier en sautant les 3 premières lignes
        df_temp.to_excel(temp_dossier + "df" + str(i) + ".xlsx", index=False, header=False, engine='openpyxl')
        i += 1

    df_final = pd.DataFrame([[]])
    j = 0
    list = []
    for path_fichier in os.listdir(temp_dossier):
        if j % 10 == 0:
            print(j, '/668')
        df_temp = pd.read_excel(temp_dossier + path_fichier)
        df_temp = df_temp.dropna(subset=["IAR", "Adresse Sissi du site PTO"], how="any")
        list.append(df_temp)
        j += 1

    df_final = pd.concat(list, axis=0)
    df_final.to_excel("C:/Users/SPML0410/Downloads/merge.xlsx", index=False)

    print("Travail terminé")


def bonnes_colonnes():
    """
    :param df: Un dataframe panda.
    :param col: Une liste de colonnes qui doivent être dans le dataframe.
    :return: Le dataframe avec les bonnes colonnes (ajoutées pour celles qui n'existaient pas et supprimées pour celles
    qui ne sont pas bonnes).
    """
    df = pd.read_excel("C:/Users/SPML0410/Downloads/merge.xlsx")
    vraies_colonnes = ["Adresse PB ou PT OT", "Adresse Sissi du site PTO", "Opérateur", "IAR"]
    colonnes_a_enlever = []
    # Pour chaque colonne dans l'ensemble des colonnes du dataframe.
    for colonne in df.columns:
        # Si la colonne fait partie des vraies colonnes.
        if colonne in vraies_colonnes:
            # On l'enlève des vraies colonnes (cad qu'il n'y a pas besoin de l'ajouter, voire après)
            vraies_colonnes.remove(colonne)
        else:
            # Sinon il faut enlever cette colonne.
            colonnes_a_enlever.append(colonne)

    taille = len(df)

    # On enlève les colonnes à enlever.
    df = df.drop(columns=colonnes_a_enlever, axis=0)

    # Pour chaque colonne de vraies_colonnes
    # (donc les colonnes que l'on veut avoir dans le dataframe mais qui n'y sont pas).
    for colonne in vraies_colonnes:
        if colonne == "No DESIGNATION":
            df[colonne] = [1 in range(0, taille)]
        else:
            df[colonne] = ["" in range(0, taille)]
    df.to_excel("C:/Users/SPML0410/Downloads/merge_propre.xlsx", index=False)
    return df


merge_rapports_pm()
bonnes_colonnes()
