# Importation
import joblib
import pandas as pd
import warnings
import sklearn as sk
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from collections import Counter
import preparation

warnings.filterwarnings('ignore')

# Import des données préparées à partir de ETI31
eti31 = pd.read_pickle("pickle/df_stat_commentaires.pkl")
eti31 = eti31.drop(columns="No DESIGNATION")


# Fonctions communes


def creation_test_train(df, label):
    """
    :param df: ensemble de données
    :param label: string représentant le nom de la colonne de label
    :return: les données de train et test avec les données et la labellisation.
    """
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)

    train_df_y = train_df[label]
    test_df_y = test_df[label]

    train_df_x = train_df.drop(label, axis=1)
    test_df_x = test_df.drop(label, axis=1)

    return train_df_x, train_df_y, test_df_x, test_df_y


def stat_modele(true, pred, modele, arrondi=3, affichage=False):
    """
    Enregistre un fichier txt contenant les informations du modèle testé.
    :param true: La vraie labéllisation.
    :param pred: La prédiction faite par le modele.
    :param modele: Nom du type de modèle.
    :param arrondi: L'arrondi des résultats.
    :param affichage: Indique si on doit afficher dans le terminal les résultats.
    :return: L'accuracy, le rappel, la spécificité et la valeur prédictive négative du modèle.
    """
    accuracy = round(sk.metrics.accuracy_score(true, pred), arrondi) * 100
    rappel = round(sk.metrics.recall_score(true, pred), arrondi) * 100
    precision = round(sk.metrics.precision_score(true, pred), arrondi) * 100
    specificity = round(sk.metrics.recall_score(true, pred, pos_label=0), arrondi) * 100
    vpn = round(sk.metrics.precision_score(true, pred, pos_label=0), arrondi) * 100
    if affichage:
        print("Accuracy : proportion de prédictions correctes (VP + VN /(VP+VN+FN+FP)) : " + str(accuracy) + "%")
        print(
            "Rappel : proportion de réussites correctement prédites sur l'ensemble des réussites (VP/(VP+FN)) : " + str(
                rappel) + "%")
        print("Précision :  proportion de véritables réussites parmi les réussites prédites (VP/(VP+FP)) : " + str(
            precision) + "%")
        print("Spécificité : proportion d'échecs correctement prédits sur l'ensemble des échecs (VN/(VN+FP)) : " + str(
            specificity) + "%")
        print(
            "Valeur prédictive négative: proportion de véritables échecs parmi les échecs prédits (VN/(VN+FN)) : "
            + str(vpn) + "%")
    fichier = open("modelisation/" + modele + ".txt", "a")
    fichier.truncate(0)
    fichier.write(
        "Accuracy : proportion de predictions correctes (VP + VN /(VP+VN+FN+FP)) : "
        + str(accuracy) + "% \n" +
        "Rappel : proportion de reussites correctement predites sur l'ensemble des reussites (VP/(VP+FN)): "
        + str(rappel) + "% \n" +
        "Specificite : proportion d'echecs correctement predits sur l'ensemble des echecs (VN/(VN+FP)) : "
        + str(specificity) + "% \n" +
        "Valeur predictive negative: proportion de veritables echecs parmi les echecs predits (VN/(VN+FN)) : "
        + str(vpn) + "% \n"
    )
    fichier.close()
    return accuracy, rappel, precision, specificity, vpn


def afficher_arbre_decision(class_names, feature_names, arbre):
    """
    Affiche l'arbre de décision
    :param class_names: liste contenant les noms des classes
    :param feature_names: liste des variables
    :param arbre: Un DecisionTreeClassifier créer avec Scikit Learn
    """
    plt.subplots(figsize=(12, 12))
    tree.plot_tree(arbre, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()


def pred_seuil(df, modele_name, seuil=0.5):
    """
    :param df: un dataframe panda pré-traité
    :param modele_name: un string indiquant le type de modèle
    :param seuil: (optionnel) seuil à partir duquel on souhaite labélliser une donnée comme un échec.
    :return: La liste correspondant à la labéllisation des données du dataframe d'après le modèle.
    """
    modele = joblib.load(
        "C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/modelisation/" + modele_name + ".joblib")
    predictions = modele.predict_proba(df)
    return [False if proba[0] >= seuil else True for proba in predictions]


def pred_probas(df, modele_name):
    """
    :param df: un dataframe panda pré-traité
    :param modele_name: un string indiquant le type de modèle
    :return: deux listes correspondant aux probabilités d'échec et de réussite des données de df.
    """
    mod = joblib.load(
        "C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/modelisation/" + modele_name + ".joblib")
    if modele_name == "svm":
        predictions = mod._predict_proba_lr(df)
    else:
        predictions = mod.predict_proba(df)
    col_false = []
    col_true = []
    for i in range(len(df)):
        col_false.append(predictions[i][0])
        col_true.append(predictions[i][1])

    return col_false, col_true


# ------------ Arbre de décision ---------------------------------------------------------------------------------------

# Fonctions de création du modele à partir d'eti31.
def best_hyperparameters_vpn_arbre(train_x, train_y, test_x, test_y, min_example, depth):
    """
    :param train_x: données de train
    :param train_y: label de train
    :param test_x: données de test
    :param test_y: label de test
    :param min_example: Liste contenant les minimums d'exemples par branche que l'on souhaite tester.
    :param depth: Liste contenant profondeur d'arbre que l'on souhaite tester.
    :return: Le meilleur vpn, le meilleur minimum d'exemples, la meilleure profondeur
             et le meilleur critère pour ces données.
    """
    best_vpn = 0
    best_min_example = 0
    best_depth = 0
    best_criterion = ""

    # On va tester toutes les combinaisons possibles :
    for criterion in ["gini", "entropy"]:
        for example in min_example:
            for profondeur in depth:
                # On crée l'arbre
                arbre = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=example, max_depth=profondeur)
                arbre.fit(train_x, train_y)
                # On calcule la prédiction et la vpn associée
                prediction_y = arbre.predict(test_x)
                vpn = round(sk.metrics.precision_score(test_y, prediction_y, pos_label=0), 5) * 100
                # On compare
                if vpn > best_vpn:
                    best_vpn = vpn
                    best_min_example = example
                    best_depth = profondeur
                    best_criterion = criterion

    return best_vpn, best_min_example, best_depth, best_criterion


def finetune_vpn_arbre(nombre_essai, min_example, depth):
    """
    :param nombre_essai: entier
    :param min_example: liste d'entiers pour le minimum d'exemple dans chaque branche de l'arbre de décision
    :param depth: liste d'entiers pour le maximum de profondeur de l'arbre
    :return: la liste des meilleurs paramètres
    """
    l_vpn = []
    l_best_min_example = []
    l_best_depth = []
    l_best_criterion = []

    for i in range(0, nombre_essai):
        if i % 10 == 0:
            print(i)
        trainx, trainy, testx, testy = creation_test_train(eti31, "Reussite")
        best_vpn, best_min_example, best_depth, best_criterion = \
            best_hyperparameters_vpn_arbre(trainx, trainy, testx, testy, min_example, depth)
        l_vpn.append(best_vpn)
        l_best_min_example.append(best_min_example)
        l_best_depth.append(best_depth)
        l_best_criterion.append(best_criterion)
    return mean(l_vpn), Counter(l_best_min_example).most_common(1)[0][0], Counter(l_best_depth).most_common(1)[0][0], \
        Counter(l_best_criterion).most_common(1)[0][0]


def modele_arbre_decision():
    """
    Crée un nouveau modèle d'arbre enregistré dans un joblib avec un fichier texte associé qui contient
    les informations statistiques.
    """
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(eti31, "Reussite")
    arbre = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, max_depth=6)
    arbre.fit(train_df_x, train_df_y)
    stat_modele(test_df_y, arbre.predict(test_df_x), "arbre_decision", affichage=True)
    joblib.dump(arbre,
                "/Projet 1 - Commentaires/commentaires/modelisation/arbre_decision.joblib")


# ------------ Arbre de décision et Boosting  --------------------------------------------------------------------------


def best_hyperparameters_vpn_boosting(train_x, train_y, test_x, test_y, min_example, depth):
    """
    :param train_x: données de train
    :param train_y: label de train
    :param test_x: données de test
    :param test_y: label de test
    :param min_example: Liste contenant les minimums d'exemples par branche que l'on souhaite tester.
    :param depth: Liste contenant profondeur d'arbre que l'on souhaite tester.
    :return: Le meilleur vpn, le meilleur minimum d'exemples, la meilleure profondeur
             et le meilleur critère pour ces données.
    """
    best_vpn = 0
    best_min_example = 0
    best_depth = 0
    best_criterion = ""
    # On va tester toutes les combinaisons possibles :
    for criterion in ["gini", "entropy"]:
        for example in min_example:
            for profondeur in depth:
                # On crée l'arbre
                arbre = RandomForestClassifier(criterion=criterion, min_samples_leaf=example, max_depth=profondeur)
                arbre.fit(train_x, train_y)
                # On calcule la prédiction et la vpn associée
                prediction_y = arbre.predict(test_x)
                vpn = round(sk.metrics.precision_score(test_y, prediction_y, pos_label=0), 5) * 100
                # On compare
                if vpn > best_vpn:
                    best_vpn = vpn
                    best_min_example = example
                    best_depth = profondeur
                    best_criterion = criterion

    return best_vpn, best_min_example, best_depth, best_criterion


def finetune_vpn_boosting(nombre_essai, min_example, depth):
    """
    :param nombre_essai: entier
    :param min_example: liste d'entiers pour le minimum d'exemple dans chaque branche de l'arbre de décision
    :param depth: liste d'entiers pour le maximum de profondeur de l'arbre
    :return: la liste des meilleurs paramètres
    """
    l_vpn = []
    l_best_min_example = []
    l_best_depth = []
    l_best_criterion = []

    for i in range(0, nombre_essai):
        print(i)
        train_x, train_y, test_x, test_y = creation_test_train(eti31, "Reussite")
        best_vpn, best_min_example, best_depth, best_criterion = \
            best_hyperparameters_vpn_boosting(train_x, train_y, test_x, test_y, min_example, depth)
        l_vpn.append(best_vpn)
        l_best_min_example.append(best_min_example)
        l_best_depth.append(best_depth)
        l_best_criterion.append(best_criterion)
    return mean(l_vpn), Counter(l_best_min_example).most_common(1)[0][0], Counter(l_best_depth).most_common(1)[0][0], \
        Counter(l_best_criterion).most_common(1)[0][0]


def modele_arbre_boosting():
    """
    Crée un nouveau modèle de forêts aléatoires (arbre_boosting) et enregistre les statistiques associées à ce modèle
    dans un fichier txt.
    """
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(eti31, "Reussite")
    arbre = RandomForestClassifier(criterion="gini", min_samples_leaf=2, max_depth=6)
    arbre.fit(train_df_x, train_df_y)
    stat_modele(test_df_y, arbre.predict(test_df_x), "arbre_boosting", affichage=True)
    joblib.dump(arbre,
                "/Projet 1 - Commentaires/commentaires/modelisation/arbre_boosting.joblib")


# ------------ Nearest Neighbors ---------------------------------------------------------------------------------------

def best_hyperparameters_vpn_nn(neighbors, weights, p):
    """
    :param neighbors: liste d'entiers du nombre de voisins que l'on souhaite tester.
    :param weights: Liste de string contenant le type de poids que l'on veut tester.
    :param p: Liste d'entier avec le type de calcul de distance.
    :return: Le meilleur vpn, le meilleur nombre de voisins, le meilleur type de calcul de distance
    et le meilleur type de calcul de distance.
    """
    best_vpn = 0
    best_n_neighbor = 0
    best_weight = 0
    best_p = 0
    train_x, train_y, test_x, test_y = creation_test_train(eti31, "Reussite")

    # On teste toutes les combinaisons possibles.

    for n_neighbor in neighbors:
        for weight in weights:
            for i in p:
                # On crée le voisinage
                neigh = KNeighborsClassifier(n_neighbors=n_neighbor, weights=weight, p=i)
                neigh.fit(train_x, train_y)
                # On calcule la VPN associée
                predictions = neigh.predict_proba(test_x)
                seuil = 0.9
                prediction_y = [False if proba[0] >= seuil else True for proba in predictions]
                vpn = round(sk.metrics.precision_score(test_y, prediction_y, pos_label=0), 5) * 100
                # On compare
                if vpn > best_vpn:
                    best_vpn = vpn
                    best_n_neighbor = n_neighbor
                    best_weight = weight
                    best_p = i
    return best_vpn, best_n_neighbor, best_weight, best_p


def finetune_vpn_nn(essais, neighbors, weights, p):
    """
    :param essais: entier du nombre d'essais
    :param neighbors: liste d'entiers du nombre de voisins que l'on souhaite tester.
    :param weights: Liste de string contenant le type de poids que l'on veut tester.
    :param p: Liste d'entier avec le type de calcul de distance.
    :return: Les meilleurs paramètres.
    """
    l_vpn = []
    l_best_n_neighbor = []
    l_best_weight = []
    l_best_p = []

    for i in range(0, essais + 1):
        if i % 100 == 0:
            print(i)
        best_vpn, best_n_neighbor, best_weight, best_p = best_hyperparameters_vpn_nn(neighbors, weights, p)
        l_vpn.append(best_vpn)
        l_best_n_neighbor.append(best_n_neighbor)
        l_best_weight.append(best_weight)
        l_best_p.append(best_p)
    return mean(l_vpn), Counter(l_best_n_neighbor).most_common(1)[0][0], Counter(l_best_weight).most_common(1)[0][0], \
        Counter(l_best_p).most_common(1)[0][0]


def modele_nn():
    """
    Crée un nouveau modèle de type nearest neighbors, l'enregistre dans un joblib et enregistre les statistiques du
    modèle dans un fichier txt.
    """
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(eti31, "Reussite")
    neigh = KNeighborsClassifier(n_neighbors=7, weights="uniform", p=1)
    neigh.fit(train_df_x, train_df_y)
    stat_modele(test_df_y, neigh.predict(test_df_x), "nearest_neighbors", arrondi=5, affichage=True)
    joblib.dump(neigh,
                "/Projet 1 - Commentaires/commentaires/modelisation/nearest_neighbors.joblib")


# ------------ Regression logistique -----------------------------------------------------------------------------------
def modele_regression_logistique():
    """
    Crée un nouveau modèle de régression logistique et l'enregistre dans un fichier joblib et
    enregistre les statistiques du modèle dans un fichier txt.
    """
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(eti31, "Reussite")
    lgr = LogisticRegression()
    lgr.fit(train_df_x, train_df_y)
    stat_modele(test_df_y, lgr.predict(test_df_x), "regression_logistique", arrondi=5, affichage=True)
    joblib.dump(lgr,
                "/Projet 1 - Commentaires/commentaires/modelisation/logistic_regression.joblib")


# ------------ SVM -----------------------------------------------------------------------------------------------------
def modele_svm():
    """
    Créer un modèle de type svm, l'enregistre dans un fichier joblib et enregistre ses statistiques dans un
    fichier texte.
    """
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(eti31, "Reussite")
    clf = svm.LinearSVC()
    clf.fit(train_df_x, train_df_y)
    stat_modele(test_df_y, clf.predict(test_df_x), "svm", arrondi=5, affichage=True)
    joblib.dump(clf,
                "/Projet 1 - Commentaires/commentaires/modelisation/svm.joblib")


# ------------ Naive Bayes ---------------------------------------------------------------------------------------------


def modele_nb():
    """
    Crée un modèle du type naive bayes, l'enregistre dans un fichier joblib et enregistre ses statistiques dans
    un fichier texte.
    """
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(eti31, "Reussite")
    mod = GaussianNB()
    mod.fit(train_df_x, train_df_y)
    stat_modele(test_df_y, mod.predict(test_df_x), "naive_bayes", arrondi=5, affichage=True)
    joblib.dump(mod,
                "/Projet 1 - Commentaires/commentaires/modelisation/naive_bayes.joblib")


# ------------ Résultats de la modélisation ----------------------------------------------------------------------------

def resultats():
    """
    Affiche un plot scatter donnant la vpn et la spécificité de 70 modèles de chaque type.
    """
    # Données sous forme de liste de [Spécificité,VPN]
    modele_type = ["arbre_decision", "arbre_boosting", "nearest_neighbors", "regression_logistique", "svm",
                   "naive_bayes"]
    df = pd.DataFrame({"Type": [], "Specificite": [], "VPN": []})
    nombre_points = 200
    for i in range(nombre_points):
        if i % 10 == 0:
            print(i)
        train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(eti31, "Reussite")
        for modele_name in modele_type:
            match modele_name:
                case "arbre_decision":
                    modele = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, max_depth=6)
                case "arbre_boosting":
                    modele = RandomForestClassifier(criterion="gini", min_samples_leaf=2, max_depth=6)
                case "nearest_neighbors":
                    modele = KNeighborsClassifier(n_neighbors=7, weights="uniform", p=1)
                case "regression_logisitque":
                    modele = LogisticRegression()
                case "svm":
                    modele = svm.LinearSVC()
                case _:
                    modele = GaussianNB()
            modele.fit(train_df_x, train_df_y)
            acc, rap, pre, spe, vpn = stat_modele(test_df_y, modele.predict(test_df_x), modele_name, arrondi=3,
                                                  affichage=False)
            new_df = pd.DataFrame({"Type": [modele_name], "Specificite": [spe],
                                   "VPN": [vpn]})
            df = pd.concat([df, new_df], axis=0)

    df.reset_index(drop=True)
    sns.lmplot(x="Specificite", y="VPN", data=df, fit_reg=False, hue='Type', legend=False)
    plt.legend(loc='lower left')
    plt.savefig("resultats")
    plt.show()


# ----------- Automatisation du calcul des résultats -------------------------------------------------------------------

def prediction(type_modele, probas, seuil=0.5, chemin_fichier=None, chemin_dossier=None):
    """
    Fonction de prédiction des échecs ou réussite.
    :param type_modele: Un string, le type de modèle que l'on souhaite utiliser pour la prédiction. Au choix :
    arbre_decision, arbre_boosting, nearest_neighbors, logistic_regression, svm ou naive_bayes
    :param probas: Booléen indiquant si le résultat est rendu sous forme de label "réussite" ou "échec"
    ou sous forme de tuple de float entre 0 et 1 correspondants à la probabilité de réussite et la probabilité d'échec.

    :param chemin_fichier: (optionnel, default=None) Chemin d'accès au fichier contenant les commentaires
    :param chemin_dossier: (optionnel, default=None) Chemin d'accès au fichier de sauvegarde des dataframes.
    :param seuil : (optionnel, default=0.5) float indiquant le seuil

    :return: Le dataframe sous forme csv avec les prédictions dans la dernière colonne.
    """

    # On fait la preparation des données avec la fonction traitement_entier du fichier python preparation.
    # La fonction renvoie le dossier dans lequel le traitement a été fait.
    # On récupère le dataframe post-traitement.
    dossier = preparation.traitement_entier(chemin_dossier=chemin_dossier,
                                            chemin_fichier=chemin_fichier, code_releve=False)
    df = pd.read_pickle(dossier + "/df_stat_commentaires.pkl")

    sauvegarde_no_designation = df["No DESIGNATION"]
    df = df.drop(columns="No DESIGNATION", axis=1)
    try:
        if probas:

            df["Proba échec"], df["Proba réussite"] = pred_probas(df, type_modele)
            print(df["Proba échec"])
            print(df["Proba réussite"])
        else:
            df["Reussite"] = pred_seuil(df, modele_name=type, seuil=seuil)
            print(df["Reussite"])
    except FileNotFoundError:
        print("Nous n'avons pas trouvé le modèle. Choisissez : arbre_decision, arbre_boosting, nearest_neighbors, "
              "logistic_regression, svm, naive_bayes")
    df["No DESIGNATION"] = sauvegarde_no_designation
    return df


"""
prediction("naive_bayes", True,
           chemin_fichier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/xlsx/faux_commentaires.xlsx",
           chemin_dossier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/pickle_faux_commentaires")



modele_nb()
modele_svm()
modele_regression_logistique()
modele_nn()
modele_arbre_boosting()
modele_arbre_decision()
print("arbre de décision ------------")
prediction("arbre_decision", True,
           chemin_fichier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/xlsx/faux_commentaires.xlsx"
           , chemin_dossier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/pickle_faux_commentaires")
print("boosting ------------------")
prediction("arbre_boosting", True,
           chemin_fichier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/xlsx/faux_commentaires.xlsx"
           , chemin_dossier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/pickle_faux_commentaires")
print("neighbors -----------------")
prediction("nearest_neighbors", True,
           chemin_fichier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/xlsx/faux_commentaires.xlsx"
           , chemin_dossier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/pickle_faux_commentaires")
print("logistic reg ---------------")
prediction("logistic_regression", True,
           chemin_fichier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/xlsx/faux_commentaires.xlsx"
           , chemin_dossier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/pickle_faux_commentaires")
print("svm -------------------------")
prediction("svm", True,
           chemin_fichier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/xlsx/faux_commentaires.xlsx"
           , chemin_dossier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/pickle_faux_commentaires")
print("naive bayes -----------------")
prediction("naive_bayes", True,
           chemin_fichier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/xlsx/faux_commentaires.xlsx"
           , chemin_dossier="C:/Users/SPML0410/Documents/MachineLearning2/projet/commentaires/pickle_faux_commentaires")
"""
resultats()
