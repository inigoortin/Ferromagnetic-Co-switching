# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 19:44:56 2025

@author: iorti
"""

# %% Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy.optimize import fmin
import random
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
palette = sns.color_palette("muted")

# %% Import files
MO = pd.read_csv("aval_MOKE.csv")
CAT = pd.read_csv("aval_MR.csv")
MO = MO.rename(columns={'ti': 't'})

#MO = pd.read_csv("aval_MOKE_B.csv")
#MO = MO.rename(columns={'ti': 't'})
#CAT = pd.read_csv("aval_MR_B.csv")

MO = pd.read_csv("aval_MOKE_N.csv")
CAT = pd.read_csv("aval_MR_N.csv")
MO = MO.rename(columns={'ti': 't'})

#MO = pd.read_csv("aval_MOKE_B_N.csv")
#MO = MO.rename(columns={'ti': 't'})
#CAT = pd.read_csv("aval_MR_B_N.csv")
# %%
CAT = CAT[['ti', 'tf', 'S', 'Archivo']]
MO = MO[['t', 'S', 'Archivo']]
CAT['T'] = CAT['tf']-CAT['ti']+1
CAT = CAT.rename(columns={"S": "S_MR"})
CAT['S_MO'] = 0.0
CAT['I'] = 0
MO['I'] = 1
rango = [i for i in range(0, 202) if i != 165] # File 165 is incorrect
#rango = range(0,182)
for i in rango:
    MO_aux = MO[MO['Archivo']==i]
    MR_aux = CAT[CAT['Archivo']==i]
    MO_aux = MO_aux.reset_index(drop=True)
    MR_aux = MR_aux.reset_index(drop=True)
    lmo = len(MO_aux)
    lmr = len(MR_aux)
    for j in range(lmr):
        S_MOj = 0
        for k in range (lmo):
            if (MR_aux['ti'][j] <= MO_aux['t'][k] <= MR_aux['tf'][j]):
                S_MOj += MO_aux['S'][k]
                MO.loc[(MO["Archivo"] == i) & (MO["t"] == MO_aux['t'][k]), "I"] = 0
        CAT.loc[(CAT["Archivo"] == i) & (CAT["ti"] == MR_aux['ti'][j]), "S_MO"] = S_MOj
MO2 = MO[MO['I']==1]
MO2 = MO2.rename(columns={"t": "ti"})
MO2 = MO2.rename(columns={"S": "S_MO"})
MO2['tf'] = MO2['ti']
MOS = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti', 'S_MO', 'Archivo','I', 'tf', 'T'])  # Empty initial row

for i in rango:
    MO_aux = MO2[MO2['Archivo']==i]
    MO_aux = MO_aux.reset_index(drop=True)
    lmo = len(MO_aux)
    for j in range(0,lmo-1):
        if MO_aux['tf'][j+1] == MO_aux['ti'][j]+1:
            MO_aux.loc[j+1, 'ti'] = MO_aux.loc[j, 'ti']
            MO_aux.loc[j+1, 'S_MO'] += MO_aux.loc[j, 'S_MO']
            MO_aux.loc[j, 'tf'] = 0
    MO_aux = MO_aux[MO_aux['tf']!=0]
    MO_aux['T'] = MO_aux['tf']-MO_aux['ti']+1
    MOS = pd.concat([MOS, MO_aux], ignore_index=True) 
MOS = MOS.iloc[1:].reset_index(drop=True)
MOS['S_MR'] = 0.0
for i in rango:
    #url = f'Datos3/hysteresis_deg_{i}.dat' 
    url = f'Datos/hysteresis_deg_{i}.dat'    
    columnas = ['Corriente', 'MOKE', 'MR']
    #columnas = ['Corriente', 'MOKE', 'Hall','MR']
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    l = len(MOS[MOS['Archivo']==i])
    for j in range(l):
        idx = MOS.index[MOS["Archivo"] == i][j]
        ti = MOS.at[idx, "ti"]
        tf = MOS.at[idx, "tf"]
        #MOS.at[idx, "S_MR"] = df.at[tf, 'MR'] - df.at[ti-1, 'MR']
        MOS.at[idx, "S_MR"] = df.iloc[tf]['MR'] - df.iloc[ti-1]['MR']

new_column_order = ['ti', 'tf', 'S_MR', 'S_MO', 'T', 'Archivo', 'I']
MOS = MOS[new_column_order]
CAT = CAT[new_column_order]

CAT = pd.concat([CAT, MOS], ignore_index=True) 

# %% Function to create predictions of avalanches
def crear_pred_m ():
    #rango = range(0,182) 
    rango = [i for i in range(0, 202) if i != 165] # file 165 is incorrect
    #columnas = ['Corriente', 'MOKE', 'Hall','MR']
    columnas = ['Corriente', 'MOKE', 'MR']
    pred_m = pd.DataFrame([[0,0,0,0,0]], columns=['ti', 'tf','MR','Intensidad', 'Archivo'])
    for j in rango:
        if j % 10 == 0:
            print(j)
        predj = []
        #url = f'Datos3/hysteresis_deg_{j}.dat'   
        url = f'Datos/hysteresis_deg_{j}.dat'   
        df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
        df['t']=range(len(df))  
        df = df[df['Corriente']<=-0.025141]
        #df = df[df['Corriente']>=0.036290]
        for i in range(df.index[0]+10,len(df['t'])):
            MRpj = df['MR'][i]
            df_aux = df[(df['t'] >= i - 10) & (df['t'] <= i)]
            
            x = df_aux['t'].values
            y1 = df_aux['MR'].values
            y2 = df_aux['MOKE'].values

            x = sm.add_constant(x)  # adds the constant
            model1 = sm.OLS(y1, x)
            results1 = model1.fit()
            
            model2 = sm.OLS(y2,x)
            results2 = model2.fit()
            b2 = results2.params[0]
            m2 = results2.params[1]
            M = m2*i+b2            
            m1 = results1.params[1]   # slope
            std_m1 = results1.bse[1]  # std slope
            
            m_ajuste = 0.035 #0.035 for dataset A and 0.016 for dataset B
            if m1+std_m1<m_ajuste:
                predj.append([i,m1,MRpj])
        l = len(predj) - 1
        k = 1 # secondary index to move from a sequence of precursors to the next one
        while l >= 1:
            if k ==1: # We save the end
                final = predj[l][0]
                MRp = predj[l][2]
                inicio = predj[l][0]
            if predj[l][0] == predj[l-1][0] + 1:
                inicio = predj[l-1][0]
                predj.pop(l)
                k = 2 # We iterate through the time index until we find the start
            else:
                subset = df.iloc[inicio-1:final, df.columns.get_loc('MR')]
                max_idx = subset.idxmax()  # Índice del valor máximo
                min_idx = subset.idxmin()
                if max_idx < min_idx:
                    intensity = subset.max() - subset.min()
                else:
                    intensity = 0.0
                new_entry = pd.DataFrame([[inicio, final, MRp,intensity, j]], columns=['ti', 'tf','MR','Intensidad', 'Archivo'])
                pred_m = pd.concat([pred_m, new_entry], ignore_index=True)
                k = 1 # We reset the index to mark a new end
            l -= 1      
    
    pred_m = pred_m.iloc[1:].reset_index(drop=True)
    return pred_m
# %%
df_pred = crear_pred_m()
df_pred['T'] = df_pred['tf']-df_pred['ti']+1

# %% Verifies the prediction. Returns an updated dataframe.

def verificar_prediccion (df, df_aval, ventana=4):
    l = len(df)
    df_aval['P'] = 0 # Existence of the precursor
    df_aval['Tp'] = 0 # Period of the precursor
    df_aval['tp'] = -1 # Time in the precursor window
    df_aval['Ip'] = 0.0 # Intensity of the precursor
    for i in range(l):
        t = df['tf'][i]
        file = df['Archivo'][i]
        df_aux = df_aval[df_aval['Archivo']== file]
        ti = df_aux['ti'].to_list()
        for taval in ti:  # Loop through each value in the list of avalanches for that file
            if t  <= taval <= t + ventana:                 
                df_aval.loc[(df_aval['ti'] == taval) & (df_aval['Archivo'] == file), 'P'] += 1
                df_aval.loc[(df_aval['ti'] == taval) & (df_aval['Archivo'] == file), 'Tp'] += df['T'][i]
                df_aval.loc[(df_aval['ti'] == taval) & (df_aval['Archivo'] == file), 'tp'] = taval - t
                df_aval.loc[(df_aval['ti'] == taval) & (df_aval['Archivo'] == file), 'Ip'] += df['Intensidad'][i]
    return df_aval

# %%
CATP = verificar_prediccion(df_pred, CAT)

# %%
CATP.to_csv("CATP_N.csv", index = False)
#CATP.to_csv("CATP_B_N.csv", index = False)
