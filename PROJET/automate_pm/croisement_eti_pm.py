import pandas as pd

# Import des DataFrames
df_eti31 = pd.read_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/automate_pm/ETI31_a_croiser.xlsx")
df_pm = pd.read_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/automate_pm/merge_propre.xlsx")
df_pm["iar_ndfictif"] = df_pm["IAR"]
df_pm = df_pm.drop(columns="IAR")

croisement = df_eti31.merge(df_pm, on="iar_ndfictif")
croisement.to_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/automate_pm/croisement.xlsx", index=False)

# On fait la jointure
test1 = pd.DataFrame({"A": [2, 3, 1], "B": ["ie", "cad", "well"]})
test2 = pd.DataFrame({"A": [1, 2, 3], "b": ["one", "two", "three"]})

test3 = test1.merge(test2, on="A")
print("kkz")
