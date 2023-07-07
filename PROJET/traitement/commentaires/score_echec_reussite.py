import pandas as pd
import warnings
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')

# On récupère le dataframe avec les données de ETI31 labellisé
df = pd.read_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/ETI31_Label_adresses_commentaires.xlsx")

# On enlève les colonnes inutiles
df = df.drop(columns=["iar_ndfictif", 'adresse_client', 'adresse_pb'], axis=0)
df = df.astype({'COMMLITIGEPIDI': str, 'COMMENTCONTACT': str, 'BLOC_NOTE': str, 'COMMENTAIRE': str, 'COMMENTAIRE_DE_SIG'
: str})


def colonne_commentaires(dataframe, col_exceptions):
    """"
    :param dataframe: Un dataframe contenant des colonnes de string à concaténer dans une seule colonne.
    :param col_exceptions: La liste des colonnes à ne pas concaténer.
    """
    liste = []
    for i in range(len(dataframe)):
        commentaires = ""
        for colonne in dataframe.columns:
            if not (colonne in col_exceptions):
                commentaires += dataframe.loc[i, colonne]
        liste.append(commentaires)
    return liste


# On concatène tous les strings dans une seule colonne commentaires.
df["commentaires"] = colonne_commentaires(df, ["Reussite"])

# Et on enlève les colonnes maintenant inutiles.
df2 = df.drop(columns=['COMMLITIGEPIDI', 'COMMENTCONTACT', 'BLOC_NOTE', 'COMMENTAIRE', 'COMMENTAIRE_DE_SIG'], axis=0)

# ----------------------------------------------------------------------------------------------------------------------
# Création d'un dataframe avec pour colonne un mot et comme ligne une donnée de df2
# Dans une cellule, il y a un entier qui correspond au nombre de fois que ce mot apparait dans la donnée.

final_stopwords_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait",
                        "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit",
                        "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez",
                        "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux",
                        "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec",
                        "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben",
                        "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci",
                        "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent",
                        "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette",
                        "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci",
                        "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit",
                        "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois",
                        "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de",
                        "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant",
                        "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit",
                        "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant",
                        "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre",
                        "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent",
                        "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi",
                        "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut",
                        "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm",
                        "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'",
                        "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à",
                        "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la",
                        "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors",
                        "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints",
                        "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm",
                        "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait",
                        "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul",
                        "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre",
                        "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant",
                        "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi",
                        "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles",
                        "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux",
                        "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept",
                        "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq",
                        "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf",
                        "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf",
                        "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze",
                        "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize",
                        "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel",
                        "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques",
                        "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'",
                        "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize",
                        "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez",
                        "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois",
                        "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit",
                        "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze",
                        "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze",
                        "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six",
                        "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons",
                        "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle",
                        "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois",
                        "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit",
                        "trente-neuf", "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très",
                        "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux",
                        "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois",
                        "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'",
                        "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées",
                        "étés", "êtes", "être", "ô"]

cv = CountVectorizer(stop_words=final_stopwords_list)
data_cv = cv.fit_transform(df2.commentaires)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names_out())
data_dtm.index = df2.index

# Récupération du nombre d'échecs et de réussites
nb_lignes = len(df2)
nb_reussite = len(df2[df2["Reussite"] == True])
nb_echec = len(df2[df2["Reussite"] == False])


def dataframe_apparition():
    """"
    # Fonction pour obtenir le dataframe contenant les statistiques de chaque mot
    # i.e. est-ce qu'ils sont plus souvent présent dans les commentaires d'échec ou de réussite avec un seuil à au moins
    # 25% de différence (essai sur 10% qui était trop faible).
    """
    new_df = pd.DataFrame({"Reussite": [True, False]})

    for mot in data_dtm.columns:
        somme_true = 0
        somme_false = 0

        # On regarde le nombre de fois que ce mot apparait dans chaque ligne.
        for i in range(0, nb_lignes):
            value = data_dtm.loc[i, mot]
            if value != 0:
                if df2.loc[i, "Reussite"] == True:
                    somme_true += value
                else:
                    somme_false += value
        # On calcule la proportion.
        prop_reussite = round(somme_true / nb_reussite, 4)
        prop_echec = round(somme_false / nb_echec, 4)

        # On le prend en compte seulement si la différence entre l'appartion dans les réussites et dans les échecs est
        # suffisament grande.
        if abs(prop_reussite - prop_echec) > 0.2:
            new_df[mot] = [prop_reussite, prop_echec]

    # On renvoie le dataframe avec pour seulement les mots qui ont différence de proportion notable.
    return new_df


df3 = dataframe_apparition()
print(len(df3.columns))
df3.to_pickle("C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/df_score_echec_reussite.pkl")
df3.to_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/traitement/df_score_echec_reussite.xlsx")
