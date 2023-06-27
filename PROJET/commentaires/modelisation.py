# Importation
import pandas as pd
import warnings
import sklearn as sk
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from collections import Counter
import preparation
from joblib import dump, load

warnings.filterwarnings('ignore')

# Import des données préparées à partir de ETI31
df2 = pd.read_pickle("pickle/df_stat_commentaires.pkl")
df2 = df2.drop(columns="No DESIGNATION")
print(df2.columns)

# Fonctions communes
def creation_test_train(dataframe, label):
    train_df = dataframe.sample(frac=0.8)
    test_df = dataframe.drop(train_df.index)

    train_df_Y = train_df[label]
    test_df_Y = test_df[label]

    train_df_X = train_df.drop(label, axis=1)
    test_df_X = test_df.drop(label, axis=1)

    return train_df_X, train_df_Y, test_df_X, test_df_Y


def stat_modele(true, pred, arrondi=5, affichage=False):
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
            "Valeur prédictive négative: proportion de véritables échecs parmi les échecs prédits (VN/(VN+FN)) : " + str(
                vpn) + "%")

    return accuracy, rappel, precision, specificity, vpn


def afficher_arbre_decision(class_names, feature_names, arbre):
    plt.subplots(figsize=(12, 12))
    tree.plot_tree(arbre, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

# ------------ Arbre de décision ---------------------------------------------------------------------------------------


def best_hyperparameters_vpn(train_x, train_y, test_x, test_y, min_example, depth):
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


def finetune_vpn(nombre_essai, min_example, depth):
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
        trainX, trainY, testX, testY = creation_test_train(df2, "Reussite")
        best_vpn, best_min_example, best_depth, best_criterion = best_hyperparameters_vpn(trainX, trainY, testX, testY,
                                                                                          min_example, depth)
        l_vpn.append(best_vpn)
        l_best_min_example.append(best_min_example)
        l_best_depth.append(best_depth)
        l_best_criterion.append(best_criterion)
    return mean(l_vpn), Counter(l_best_min_example).most_common(1)[0][0], Counter(l_best_depth).most_common(1)[0][0], \
        Counter(l_best_criterion).most_common(1)[0][0]


def meilleur_arbre(df):
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(df, "Reussite")
    arbre = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, max_depth=6)
    arbre.fit(train_df_x, train_df_y)
    print(arbre)
    class_names = ["Réussite", "Echec"]
    features_names = ["score_reussite", "score_echec", "nombre_commentaires", "Longueur moyenne commentaire",
                      "polarite", "subjectivite"]
    afficher_arbre_decision(class_names, features_names, arbre)


def arbre_decision():
    print("Résultats de l'optimisation",finetune_vpn(250, [2, 3], [2, 3, 4, 5, 6]))
    meilleur_arbre(df2)

# Le seuil


def arbre_seuil():
    train_df_x, train_df_y, test_df_x, test_df_y = creation_test_train(df2, "Reussite")
    arbre2 = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, max_depth=6)
    arbre2.fit(train_df_x, train_df_y)

    predictions = arbre2.predict_proba(test_df_x)
    seuil = 1
    predict_label_seuil = [False if proba[0] >= seuil else True for proba in predictions]
    stat_modele(test_df_y, predict_label_seuil, arrondi=5, affichage=True)


# ------------ Arbre de décision et Boosting  --------------------------------------------------------------------------


def BestHyperparameters_vpn_boosting(trainX, trainY, testX, testY, min_example, depth):
    best_vpn = 0
    best_min_example = 0
    best_depth = 0
    best_criterion = ""
    # On va tester toutes les combinaisons possibles :
    for criterion in ["gini", "entropy"]:
        for example in min_example:
            for profondeur in depth:
                # On créer l'arbre
                arbre = RandomForestClassifier(criterion=criterion, min_samples_leaf=example, max_depth=profondeur)
                arbre.fit(trainX, trainY)
                # On calcule la prédiction et la vpn associée
                predictionY = arbre.predict(testX)
                vpn = round(sk.metrics.precision_score(testY, predictionY, pos_label=0), 5) * 100
                # On compare
                if vpn > best_vpn:
                    best_vpn = vpn
                    best_min_example = example
                    best_depth = profondeur
                    best_criterion = criterion

    return (best_vpn, best_min_example, best_depth, best_criterion)


def finetune_vpn_boosting(df, nombre_essai, min_example, depth):
    l_vpn = []
    l_best_min_example = []
    l_best_depth = []
    l_best_criterion = []

    for i in range(0, nombre_essai):
        print(i)
        trainX, trainY, testX, testY = creation_test_train(df, "Reussite")
        best_vpn, best_min_example, best_depth, best_criterion = \
            BestHyperparameters_vpn_boosting(trainX, trainY, testX, testY, min_example, depth)
        l_vpn.append(best_vpn)
        l_best_min_example.append(best_min_example)
        l_best_depth.append(best_depth)
        l_best_criterion.append(best_criterion)
    return mean(l_vpn), Counter(l_best_min_example).most_common(1)[0][0], Counter(l_best_depth).most_common(1)[0][0], \
        Counter(l_best_criterion).most_common(1)[0][0]


def meilleur_arbre_boosting(df):
    return
def arbre_decision_boosting(df):
    print("Résultats de l'optimisation",finetune_vpn_boosting(df, 250, [2, 3], [2, 3, 4, 5, 6]))
    meilleur_arbre_boosting(df2)


def arbre_seuil_boosting(df):
    train_df_X, train_df_Y, test_df_X, test_df_Y = creation_test_train(df, "Reussite")
    arbre = RandomForestClassifier(criterion="gini", min_samples_leaf=2, max_depth=6, random_state=42)
    arbre.fit(train_df_X, train_df_Y)
    predictions = arbre.predict_proba(test_df_X)
    seuil = 0.8
    predict_label_seuil = [False if proba[0]>=seuil else True for proba in predictions]
    stat_modele(test_df_Y, predict_label_seuil,arrondi=5, affichage=True)
    return arbre

# ------------ Nearest Neighbors ---------------------------------------------------------------------------------------

# ------------ Regression logistique -----------------------------------------------------------------------------------

# ------------ SVM -----------------------------------------------------------------------------------------------------

# ------------ Naive Bayes ---------------------------------------------------------------------------------------------

# ------------ Résultats de la modélisation ----------------------------------------------------------------------------

def resultats():
    # Données sous forme de liste de [Spécificité,VPN]
    arbre_decision = pd.DataFrame({"Type": ["arbre_decision" for x in range(0, 10)],
                                   "Specificite": [0.17, 0.15, 0.26, 0.19, 0.19, 0.17, 0.34, 0.26, 0.33, 0.27],
                                   "VPN": [0.94, 0.93, 0.96, 0.76, 0.76, 0.75, 0.87, 0.92, 0.88, 0.81]})
    random_forest = pd.DataFrame({"Type": ["random_forest" for x in range(0, 10)],
                                  "Specificite": [0.25, 0.28, 0.24, 0.23, 0.25, 0.23, 0.21, 0.2, 0.16, 0.19],
                                  "VPN": [0.95, 0.96, 1, 0.95, 1, 0.91, 1, 0.91, 1, 0.95]})
    neighbors = pd.DataFrame({"Type": ["neighbors" for x in range(0, 10)],
                              "Specificite": [0.3, 0.304, 0.371, 0.385, 0.382, 0.356, 0.367, 0.367, 0.32, 0.36],
                              "VPN": [0.97, 0.971, 0.973, 0.918, 0.929, 0.923, 0.9, 0.9, 0.94, 0.97]})
    regression = pd.DataFrame(
        {"Type": ["regression" for x in range(0, 5)], "Specificite": [0.39, 0.42, 0.36, 0.4, 0.36],
         "VPN": [0.73, 0.61, 0.62, 0.55, 0.59]})
    SVM = pd.DataFrame({"Type": ["svm" for x in range(0, 5)], "Specificite": [0.45, 0.49, 0.51, 0.54, 0.55],
                        "VPN": [0.64, 0.67, 0.64, 0.66, 0.64]})
    naive_bayes = pd.DataFrame(
        {"Type": ["naive_bayes" for x in range(0, 5)], "Specificite": [0.82, 0.78, 0.76, 0.8, 0.79],
         "VPN": [0.46, 0.46, 0.43, 0.42, 0.53]})

    df = pd.concat([arbre_decision, random_forest, neighbors, regression, SVM, naive_bayes])
    df.reset_index(drop=True)

    sns.lmplot(x="Specificite", y="VPN", data=df, fit_reg=False, hue='Type', legend=False)
    plt.legend(loc='upper right')
    plt.show()


resultats()

# ----------- Automatisation du calcul des résultats -------------------------------------------------------------------

def prediction(selection, chemin_fichier=None, chemin_dossier=None):
    dossier = preparation.traitement_entier(selection=selection)
    df = pd.read_pickle(dossier + "df_stat_commentaires.pkl")
    df = df.drop(columns = "No DESIGNATION", axis=1)

    essai = 0
    chargement = True
    limite = 3
    while chargement and essai <= limite:
        essai += 1
        try:
            type = str(input())
            modele = load(type + "_model_saved.joblib")
            chargement = True
        except ValueError:
            print("Nous n'avons pas trouvé le modèle. Choisissez : arbre_decision, arbre_boosting,"
                  " nearest_neighbors, logistic_regression, svm, naive_bayes")
    if essai <= limite:
        modele.predict(df)

def prediction_seuil(selection, seuil, chemin_fichier=None, chemin_dossier=None):
    return
