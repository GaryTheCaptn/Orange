import pandas as pd

df = pd.read_excel("C:/Users/SPML0410/Documents/MachineLearning2/projet/Projet 3 - Type PB/eti31.xlsx")
info_pb = df["info_pb"]


def statistiques():
    type_pb = []
    type_materielPB = []
    type_materialPB = []
    typePBO = []
    typeRaccoPBOPTO = []
    RaccordementLong = []
    hauteurPBO = []
    adduction = []

    for cell in info_pb:
        if "Type PB" in cell:
            type_pb.append(cell)
        if "TypeMaterielPBO" in cell:
            type_materielPB.append(cell)
        if "TypeMaterialPBO" in cell:
            type_materialPB.append(cell)
        if "TypePBO" in cell:
            typePBO.append(cell)
        if "TypeRaccoPBPTO" in cell:
            typeRaccoPBOPTO.append(cell)
        if "RaccordementLong" in cell:
            RaccordementLong.append(cell)
        if "HauteurPBO" in cell:
            hauteurPBO.append(cell)
        if "Adduction" in cell:
            adduction.append(cell)
    print("Type PB", len(type_pb), "TypeMaterielPBO", len(type_materielPB), "TypeMaterialPBO", len(type_materialPB),
          "TypePBO", len(typePBO), "TypeRaccoPBPTO", len(typeRaccoPBOPTO), "RaccordementLong", len(
            RaccordementLong), "HauteurPBO", len(hauteurPBO), "Adduction", len(adduction))


statistiques()


def statistiques2():
    typePB = []
    plus = []

    for cell in info_pb:
        if "Type PB" in cell or "TypeMaterielPBO" in cell or "TypeMaterialPBO" in cell or "TypePBO" in cell:
            typePB.append(cell)
        if "Adduction" in cell and "RaccordementLong" in cell:
            plus.append(cell)
    print("Type PB ", len(typePB), "Plus", len(plus))


statistiques2()
