# -*- coding: utf-8 -*-
"""
@author: Fernando Vallejo Banegas
@author: Luis Jiménez Navajas
"""
import pandas as pd
import numpy as np

# Leemos dataset de datos de las jugadoras durante los partidos
dataset_jugadoras = pd.read_excel("../Data/Handball Woman European Data Set.xlsx",index_col=0)

# Leemos dataset de resultados de los partidos
dataset_partidos_train = pd.read_csv("../Data/dataset_partidos.csv",delimiter=";")
matches_train = []
for i in range(len(dataset_partidos_train)):
    matches_train.append(f"{dataset_partidos_train.iloc[i]['Equipo1']}-{dataset_partidos_train.iloc[i]['Equipo2']}".replace(" ",""))

idx_test = [x for x in range(len(dataset_jugadoras)) if not (dataset_jugadoras['Match'][x] in matches_train)]
idx_train = [x for x in range(len(dataset_jugadoras)) if dataset_jugadoras['Match'][x] in matches_train]
dataset_jugadoras_train = dataset_jugadoras.drop(idx_test)
dataset_jugadoras_test = dataset_jugadoras.drop(idx_train)
cp_dataset_jugadoras_train = dataset_jugadoras_train.copy()

dataset_jugadoras_train = dataset_jugadoras_train.drop(['Phase', 'Match', 'Name','No','RC',
                                            '2+2','scoring','%','7m%',
                                            'Goals','Shots','MVP','FTOGoals',
                                            'FTOMissed','YC','Time'],axis=1)

######################### PRUEBA ELIMINAR VARIABLES CORRELACIONADAS ##########################
dataset_jugadoras_train["cor_vars_mean"] = 0
for i,elem in dataset_jugadoras_train.iterrows():
    mean = (elem.loc["AS"] + elem.loc["BS"] + elem.loc["BTMissed"] + elem.loc["BTGoals"] + elem.loc["FBGoals"] + elem.loc["FBMissed"]) / 5
    dataset_jugadoras_train.loc[i,"cor_vars_mean"] = mean
dataset_jugadoras_train = dataset_jugadoras_train.drop(['AS','BS','BTMissed','BTGoals','FBGoals','FBMissed'],axis=1)
##############################################################################################

pivdata = pd.pivot_table(dataset_jugadoras_train, index = ['Team'], aggfunc = np.sum)
pivdata['rcvGoals'] = 0

# Leemos dataset de validacion
dataset_validacion = pd.read_csv("../Data/dataset_partidos_prediccion.csv", delimiter=";")

# Calculamos el numero de goles recibidos en cada equipo
dataset_goles_recibidos = cp_dataset_jugadoras_train.groupby(by=['Match','Team'])['Goals'].sum()

idx_list = []
for i in range(len(dataset_goles_recibidos)):
    scoring_team = dataset_goles_recibidos.index[i][1]
    # To avoid KeyError
    if scoring_team == "SRB ":
        scoring_team = "SRB"
        
    receiver_team = dataset_goles_recibidos.index[i][0].replace(scoring_team,"").replace("-","")
    
    # To avoid KeyError
    if receiver_team == "SRB":
        receiver_team = "SRB "
    pivdata.loc[receiver_team,'rcvGoals'] += dataset_goles_recibidos[i]

# Calculamos el dataset de test
dataset_partidos_test = dataset_jugadoras_test.groupby(by=['Match','Team'])['Goals'].sum()

groups = dataset_partidos_test.groupby('Match')
datos = []
for x in groups.groups.keys():
    if groups.get_group(x)[0] > groups.get_group(x)[1]:
        resultado = 0
    elif groups.get_group(x)[0] == groups.get_group(x)[1]:
        resultado = 1
    elif groups.get_group(x)[0] < groups.get_group(x)[1]:
        resultado = 2
    # Crear dataframe y añadir cada fila groups.get_group(x)
    datos.append([groups.get_group(x).index[0][1],groups.get_group(x).index[1][1],resultado])
dataset_partidos_test = pd.DataFrame(datos,columns=['Equipo1','Equipo2','Resultado'])

# NO HACE FALTA, TODOS HAN JUGADO EL MISMO TIEMPO (3 PARTIDOS) -
# Normalizamos los datos en base al tiempo jugado de cada 
#pivdata = pivdata.loc[:,"2M":"rcvGoals"].div(pivdata["Time"]/16,axis=0)
#print(pivdata['Time'])
#pivdata = pivdata.drop(['Time'],axis=1)
# Join de los dataframes
dataset_df_train = dataset_partidos_train.join(pivdata, on="Equipo1")
dataset_df_train = dataset_df_train.join(pivdata, on="Equipo2", rsuffix="_Equipo2", lsuffix="_Equipo1")
dataset_df_train.to_csv('../Data/train.csv',index=False)

dataset_df_validacion = dataset_validacion.join(pivdata, on="Equipo1")
dataset_df_validacion = dataset_df_validacion.join(pivdata, on="Equipo2", rsuffix="_Equipo2", lsuffix="_Equipo1")
dataset_df_validacion.to_csv('../Data/valid.csv',index=False)

dataset_df_test = dataset_partidos_test.join(pivdata, on="Equipo1")
dataset_df_test = dataset_df_test.join(pivdata, on="Equipo2", rsuffix="_Equipo2", lsuffix="_Equipo1")
dataset_df_test.to_csv('../Data/test.csv',index=False)