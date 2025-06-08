# %% Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import fmin

import random
import matplotlib.pyplot as plt
import scienceplots # unified style and LaTex format
plt.style.use('science')
palette = sns.color_palette("muted")

# %% Avalanches Catalogues
# %%% Dataset A
# %%%% Fisrt version
# Avalanches in MOKE, T=1. Avalanches in MR, T>1.

columnas = ['Corriente', 'MOKE', 'MR']
aval_MR = pd.DataFrame(columns=['MR', 'Corriente', 'dMR', 'ti', 'tf', 'T', 'S', 'Archivo'])
aval_MOKE = pd.DataFrame(columns=['MOKE', 'Corriente', 'dMOKE', 't', 'S', 'Archivo'])
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#rango = range(0,181) 

for i in rango:
    url = f'Data_A/hysteresis_deg_{i}.dat'    
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['dMOKE'] = df['MOKE'].diff(-1) / df['Corriente'].diff(-1)
    df['dMR'] = df['MR'].diff(-1) / df['Corriente'].diff(-1)
    l = len(df)
    df['dMR'][l-1]=0
    df['dMOKE'][l-1]=0
    df['t']=range(len(df))
    
    # Canal MR
    df['MR_Filtro']=abs(df['dMR'])>50000
    df_MRf = df[df['MR_Filtro'] == True] # f de filtro
    mask = (df_MRf['t'].diff(-1) == -1) | (df_MRf['t'].diff(1) == 1)
    df_MRf2 = df_MRf[mask]
    mask2 = df_MRf2['t'].diff(1) != 1
    df_MRa = df_MRf2[mask2] # a de avalanchas

    # Añadimos el final y la duración de la avalancha así como su duración
    mask3 = df_MRf2['t'].diff(-1) != -1
    Tf_MRa = df_MRf2.loc[mask3, 't'].tolist() # .loc[row_selector, column_selector]
    df_MRa = df_MRa.copy() # Sino sale un warning al hacer la siguiente asignación de la columna
    df_MRa['tf'] = Tf_MRa
    df_MRa.rename(columns={'t': 'ti'}, inplace=True)
    df_MRa['T'] = df_MRa['tf']-df_MRa['ti']+1 # La columna T es la duración de la avalancha
    df_MRa['S'] = df_MRa.apply(lambda row: df.loc[df['t'] == row['tf'], 'MR'].values[0] - df.loc[df['t'] == row['ti'], 'MR'].values[0], axis=1)
    df_MRa = df_MRa[['MR', 'Corriente', 'dMR','ti', 'tf', 'T', 'S']] # MR, Corriente y dMR hacen referncia al comienzo de la avalancha
    df_MRa = df_MRa[df_MRa['S']>0]
    df_MRa['Archivo']=i
    aval_MR = pd.concat([aval_MR, df_MRa], ignore_index=True)
    
    # Canal MOKE
    ventana = 25 # da mejor resultado que 20 (discrimina mejor)
    df['MO_std_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
    df['MO_media_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
    df['MOKE_umbral'] =2.5*df['MO_std_local'] - df['MO_media_local'] #IMP
    df['MOKE_Filtro']=df['dMOKE']<-df['MOKE_umbral']
    df_MOf = df[df['MOKE_Filtro'] == True] # f de filtro
    df_MOf = df_MOf.copy() # evita warning
    df_MOf['S'] = df_MOf.apply(lambda row: df.loc[df['t'] == row['t']+1, 'MOKE'].values[0] - df.loc[df['t'] == row['t'], 'MOKE'].values[0], axis=1)
    df_MOf = df_MOf[['MOKE', 'Corriente', 'dMOKE', 't', 'S']]
    df_MOf['Archivo']=i
    aval_MOKE = pd.concat([aval_MOKE, df_MOf], ignore_index=True)
    
# Guardar archivos
# aval_MR.to_csv('aval_MR.csv', index=False)
# aval_MOKE.to_csv('aval_MOKE.csv', index=False)
# %%%%  Second version
# Avalanches T>=1
# Fix thresholds fo MR, commented lines for a dynamic threshold

columnas = ['Corriente', 'MOKE', 'MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo'])
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo

for i in rango:
    url = f'Data_A/hysteresis_deg_{i}.dat'    
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['dMOKE'] = df['MOKE'].diff(-1) / df['Corriente'].diff(-1)
    df['dMR'] = df['MR'].diff(-1) / df['Corriente'].diff(-1)
    l = len(df)
    df['dMR'][l-1]=0
    df['dMOKE'][l-1]=0
    df['t']=range(len(df))
    
    # Canal MR - umbral fijo
    df['MR_Filtro']=abs(df['dMR'])>30000
    df_MRf = df[df['MR_Filtro'] == True].copy() # f de filtro
    df_MRf = df_MRf.reset_index(drop=True)
    
    # Canal MR - umbral variable
    #ventana = 25 # da mejor resultado que 20 (discrimina mejor)
    #df['MR_std_local'] = df['dMR'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
    #df['MR_media_local'] = df['dMR'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
    #df['MR_umbral'] =2*df['MR_std_local'] - df['MR_media_local'] #IMP
    #df['MR_Filtro']=df['dMR']<-df['MR_umbral']
    #df_MRf = df[df['MR_Filtro'] == True].copy()# f de filtro
    #df_MRf = df_MRf.reset_index(drop=True)

    df_MRf['N'] = 0
    for j in range (1, len(df_MRf)):
        if df_MRf['t'].loc[j] == df_MRf['t'].loc[j-1]+1:
            df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']
        else: df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']+1
    LN = df_MRf['N'].unique()
    for k in LN:
        dfk = df_MRf[df_MRf['N']==k].reset_index(drop=True)
        tik = dfk['t'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk-tik+1
        Sk = df['MR'][tfk+1]- df['MR'][tik]
        new_entry_MR1 = pd.DataFrame([[tik, tfk, Tk, Sk, i]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
        aval_MR1 = pd.concat([aval_MR1, new_entry_MR1], ignore_index=True)
    
    # Canal MOKE
    ventana = 25 # da mejor resultado que 20 (discrimina mejor)
    df['MO_std_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
    df['MO_media_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
    df['MOKE_umbral'] =2.5*df['MO_std_local']+ df['MO_media_local'] #IMP
    df['MOKE_Filtro']=df['dMOKE']<-df['MOKE_umbral']
    df_MOf = df[df['MOKE_Filtro'] == True].copy()# f de filtro
    df_MOf = df_MOf.reset_index(drop=True)
    
    df_MOf['N'] = 0
    for j in range (1, len(df_MOf)):
        if df_MOf['t'].loc[j] == df_MOf['t'].loc[j-1]+1:
            df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']
        else: df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']+1
    LNO = df_MOf['N'].unique()
    for k in LNO:
        dfk = df_MOf[df_MOf['N']==k].reset_index(drop=True)
        tik = dfk['t'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk-tik+1
        Sk = df['MOKE'][tfk+1]- df['MOKE'][tik]
        new_entry_MO1 = pd.DataFrame([[tik, tfk, Tk, Sk, i]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
        aval_MOKE1 = pd.concat([aval_MOKE1, new_entry_MO1], ignore_index=True)
aval_MR = aval_MR1.iloc[1:].reset_index(drop=True)
aval_MOKE = aval_MOKE1.iloc[1:].reset_index(drop=True)

aval_MR = aval_MR[aval_MR['S'] > 0].reset_index(drop=True)
aval_MOKE = aval_MOKE[aval_MOKE['S'] > 0].reset_index(drop=True)

# Guardar archivos
#aval_MR.to_csv('aval_MR.csv', index=False)
#aval_MOKE.to_csv('aval_MOKE_N.csv', index=False)

# aval_MR contiene los datos obtenidos con umbral fijo (30000)
# %%%%  Actual version
# MOKE Standardization (0,1)

columnas = ['Corriente', 'MOKE', 'MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo', 'Corriente'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo', 'Corriente'])
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#rango = range(0,181)

for i in rango:
    url = f'Data_A/hysteresis_deg_{i}.dat'    
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    #df = df[df['Corriente']<-0.026]
    df['dMOKE'] = df['MOKE'].diff(-1) / df['Corriente'].diff(-1)
    df['dMR'] = df['MR'].diff(-1) / df['Corriente'].diff(-1)
    minM = min(df[df['Corriente']<=-0.025141]['MOKE'])
    df['M'] = (df['MOKE'] - minM) / (df['MOKE'].max() - minM)
    l = len(df)
    df['dMR'][l-1]=0
    df['dMOKE'][l-1]=0
    df['t']=range(len(df))
    
    # Canal MR - umbral fijo
    df['MR_Filtro']=abs(df['dMR'])>30000
    df_MRf = df[df['MR_Filtro'] == True].copy() # f de filtro
    df_MRf = df_MRf.reset_index(drop=True)
    
    # Canal MR - umbral variable
    #ventana = 25
    #df['MR_std_local'] = df['dMR'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
    #df['MR_media_local'] = df['dMR'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
    #df['MR_umbral'] =2*df['MR_std_local'] - df['MR_media_local'] #IMP
    #df['MR_Filtro']=df['dMR']<-df['MR_umbral']
    #df_MRf = df[df['MR_Filtro'] == True].copy()# f de filtro
    #df_MRf = df_MRf.reset_index(drop=True)

    df_MRf['N'] = 0
    for j in range (1, len(df_MRf)):
        if df_MRf['t'].loc[j] == df_MRf['t'].loc[j-1]+1:
            df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']
        else: df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']+1
    LN = df_MRf['N'].unique()
    for k in LN:
        dfk = df_MRf[df_MRf['N']==k].reset_index(drop=True)
        cik = dfk['Corriente'].loc[0]
        tik = dfk['t'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk-tik+1
        Sk = df['MR'][tfk+1]- df['MR'][tik]
        new_entry_MR1 = pd.DataFrame([[tik, tfk, Tk, Sk, i,cik]], columns=['ti', 'tf', 'T', 'S', 'Archivo', 'Corriente'])
        aval_MR1 = pd.concat([aval_MR1, new_entry_MR1], ignore_index=True)
    
    # MOKE
    ventana = 25 # da mejor resultado que 20 (discrimina mejor)
    df['MO_std_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
    df['MO_media_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
    df['MOKE_umbral'] =2.5*df['MO_std_local']+ df['MO_media_local'] #IMP
    df['MOKE_Filtro']=df['dMOKE']<-df['MOKE_umbral']
    df_MOf = df[df['MOKE_Filtro'] == True].copy()# f de filtro
    df_MOf = df_MOf.reset_index(drop=True)
    
    df_MOf['N'] = 0
    for j in range (1, len(df_MOf)):
        if df_MOf['t'].loc[j] == df_MOf['t'].loc[j-1]+1:
            df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']
        else: df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']+1
    LNO = df_MOf['N'].unique()
    for k in LNO:
        dfk = df_MOf[df_MOf['N']==k].reset_index(drop=True)
        tik = dfk['t'].loc[0]
        cik = dfk['Corriente'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk-tik+1
        Sk = df['M'][tfk+1]- df['M'][tik]
        new_entry_MO1 = pd.DataFrame([[tik, tfk, Tk, Sk, i,cik]], columns=['ti', 'tf', 'T', 'S', 'Archivo','Corriente'])
        aval_MOKE1 = pd.concat([aval_MOKE1, new_entry_MO1], ignore_index=True)
aval_MR = aval_MR1.iloc[1:].reset_index(drop=True)
aval_MOKE = aval_MOKE1.iloc[1:].reset_index(drop=True)

aval_MR = aval_MR[aval_MR['S'] > 0].reset_index(drop=True)
aval_MOKE = aval_MOKE[aval_MOKE['S'] > 0].reset_index(drop=True)

aval_MR = aval_MR[aval_MR['Corriente']<=-0.025141]
aval_MOKE = aval_MOKE[aval_MOKE['Corriente']<=-0.025141]

# Save files
aval_MR.to_csv('aval_MR_N.csv', index=False)
aval_MOKE.to_csv('aval_MOKE_N.csv', index=False)
# %%% Dataset B
# %%%% First version

columnas = ['Corriente', 'MOKE', 'Hall','MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo'])
rango = range(0,182)
for i in rango:
    url = f'Data_B/hysteresis_deg_{i}.dat'    
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['dMOKE'] = df['MOKE'].diff(-1) / df['Corriente'].diff(-1)
    df['dMR'] = df['MR'].diff(-1) / df['Corriente'].diff(-1)
    l = len(df)
    df['dMR'][l-1]=0
    df['dMOKE'][l-1]=0
    df['t']=range(len(df))
    
    # Canal MR
    df['MR_Filtro']=abs(df['dMR'])>10000
    df_MRf = df[df['MR_Filtro'] == True].copy() # f de filtro
    df_MRf = df_MRf.reset_index(drop=True)

    df_MRf['N'] = 0
    for j in range (1, len(df_MRf)):
        if df_MRf['t'].loc[j] == df_MRf['t'].loc[j-1]+1:
            df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']
        else: df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']+1
    LN = df_MRf['N'].unique()
    for k in LN:
        dfk = df_MRf[df_MRf['N']==k].reset_index(drop=True)
        tik = dfk['t'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk-tik+1
        Sk = df['MR'][tfk+1]- df['MR'][tik]
        new_entry_MR1 = pd.DataFrame([[tik, tfk, Tk, Sk, i]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
        aval_MR1 = pd.concat([aval_MR1, new_entry_MR1], ignore_index=True)
    
    # Canal MOKE
    ventana = 25 # da mejor resultado que 20 (discrimina mejor)
    df['MO_std_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
    df['MO_media_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
    df['MOKE_umbral'] =2.5*df['MO_std_local']+ df['MO_media_local'] #IMP
    df['MOKE_Filtro']=df['dMOKE']<-df['MOKE_umbral']
    df_MOf = df[df['MOKE_Filtro'] == True].copy()# f de filtro
    df_MOf = df_MOf.reset_index(drop=True)
    
    df_MOf['N'] = 0
    for j in range (1, len(df_MOf)):
        if df_MOf['t'].loc[j] == df_MOf['t'].loc[j-1]+1:
            df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']
        else: df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']+1
    LNO = df_MOf['N'].unique()
    for k in LNO:
        dfk = df_MOf[df_MOf['N']==k].reset_index(drop=True)
        tik = dfk['t'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk+1-tik
        Sk = df['MOKE'][tfk+1] - df['MOKE'][tik]
        new_entry_MO1 = pd.DataFrame([[tik, tfk, Tk, Sk, i]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
        aval_MOKE1 = pd.concat([aval_MOKE1, new_entry_MO1], ignore_index=True)
aval_MRN = aval_MR1.iloc[1:].reset_index(drop=True)
aval_MOKEN = aval_MOKE1.iloc[1:].reset_index(drop=True)

aval_MRN = aval_MRN[aval_MRN['S'] > 0].reset_index(drop=True)
aval_MOKEN = aval_MOKEN[aval_MOKEN['S'] < 0].reset_index(drop=True)
#aval_MOKEN = aval_MOKEN[aval_MOKEN['S'] > 0].reset_index(drop=True)

# Guardar archivos
aval_MRN.to_csv('aval_MR_B.csv', index=False)
aval_MOKEN.to_csv('aval_MOKE_B.csv', index=False)

# aval_MR_B contiene los Data_A obtenidos con umbral fijo (10000)

# %%%% Actual version 
# MOKE Standardization (0,1)

columnas = ['Corriente', 'MOKE', 'Hall','MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo', 'Corriente'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo','Corriente'])
rango = range(0,182)
for i in rango:
    url = f'Data_B/hysteresis_deg_{i}.dat'    
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['dMOKE'] = df['MOKE'].diff(-1) / df['Corriente'].diff(-1)
    df['dMR'] = df['MR'].diff(-1) / df['Corriente'].diff(-1)
    maxM = max(df[df['Corriente']>=0.036290]['MOKE'])
    df['M'] = (df['MOKE'] - df['MOKE'].min()) / (maxM - df['MOKE'].min())
    l = len(df)
    df['dMR'][l-1]=0
    df['dMOKE'][l-1]=0
    df['t']=range(len(df))
    
    # Canal MR
    df['MR_Filtro']=abs(df['dMR'])>10000
    df_MRf = df[df['MR_Filtro'] == True].copy() # f de filtro
    df_MRf = df_MRf.reset_index(drop=True)

    df_MRf['N'] = 0
    for j in range (1, len(df_MRf)):
        if df_MRf['t'].loc[j] == df_MRf['t'].loc[j-1]+1:
            df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']
        else: df_MRf.at[j, 'N'] = df_MRf.at[j-1, 'N']+1
    LN = df_MRf['N'].unique()
    for k in LN:
        dfk = df_MRf[df_MRf['N']==k].reset_index(drop=True)
        cik = dfk['Corriente'].loc[0]
        tik = dfk['t'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk-tik+1
        Sk = df['MR'][tfk+1]- df['MR'][tik]
        new_entry_MR1 = pd.DataFrame([[tik, tfk, Tk, Sk, i,cik]], columns=['ti', 'tf', 'T', 'S', 'Archivo','Corriente'])
        aval_MR1 = pd.concat([aval_MR1, new_entry_MR1], ignore_index=True)
    
    # Canal MOKE
    ventana = 25 # da mejor resultado que 20 (discrimina mejor)
    df['MO_std_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
    df['MO_media_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
    df['MOKE_umbral'] =2.5*df['MO_std_local']+ df['MO_media_local'] #IMP
    df['MOKE_Filtro']=df['dMOKE']<-df['MOKE_umbral']
    df_MOf = df[df['MOKE_Filtro'] == True].copy()# f de filtro
    df_MOf = df_MOf.reset_index(drop=True)
    
    df_MOf['N'] = 0
    for j in range (1, len(df_MOf)):
        if df_MOf['t'].loc[j] == df_MOf['t'].loc[j-1]+1:
            df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']
        else: df_MOf.at[j, 'N'] = df_MOf.at[j-1, 'N']+1
    LNO = df_MOf['N'].unique()
    for k in LNO:
        dfk = df_MOf[df_MOf['N']==k].reset_index(drop=True)
        tik = dfk['t'].loc[0]
        cik = dfk['Corriente'].loc[0]
        tfk = dfk['t'].iloc[-1]
        Tk = tfk+1-tik
        Sk = df['M'][tfk+1] - df['M'][tik]
        new_entry_MO1 = pd.DataFrame([[tik, tfk, Tk, Sk, i,cik]], columns=['ti', 'tf', 'T', 'S', 'Archivo','Corriente'])
        aval_MOKE1 = pd.concat([aval_MOKE1, new_entry_MO1], ignore_index=True)
aval_MRN = aval_MR1.iloc[1:].reset_index(drop=True)
aval_MOKEN = aval_MOKE1.iloc[1:].reset_index(drop=True)

aval_MRN = aval_MRN[aval_MRN['S'] > 0].reset_index(drop=True)
aval_MOKEN = aval_MOKEN[aval_MOKEN['S'] < 0].reset_index(drop=True)
#aval_MOKEN = aval_MOKEN[aval_MOKEN['S'] > 0].reset_index(drop=True)

aval_MRN = aval_MRN[aval_MRN['Corriente']>=0.036290]
aval_MOKEN = aval_MOKEN[aval_MOKEN['Corriente']>=0.036290]

# Guardar archivos
aval_MRN.to_csv('aval_MR_B_N.csv', index=False)
aval_MOKEN.to_csv('aval_MOKE_B_N.csv', index=False)

# %%% Load catalogues

aval_MR = pd.read_csv('aval_MR_N.csv')
aval_MOKE = pd.read_csv('aval_MOKE_N.csv')

aval_MR_B = pd.read_csv('aval_MR_B_N.csv')
aval_MOKE_B = pd.read_csv('aval_MOKE_B_N.csv')

# %% Power laws
# %%% Probability Density Function

def pdf_power_law (data, a, b, alpha):
    '''
    
    Parameters
    ----------
    data : TYPE
        x values.
    a : TYPE
        Lower threshold for power-law estimation.
    b : TYPE
        Upper threshold for power-law estimation.
    alpha : TYPE
        Alpha parameter.

    Returns
    -------
    y : TYPE
        Probability density fucntion values.

    '''
    y = (alpha-1)/(a**(1-alpha)-1/(b**(alpha-1)))*((1/data)**alpha)
    return y

def plot_power_law(data, xmin, xmax, alpha, titulo, pdf='prueba', nb=10000):
    '''

    Parameters
    ----------
    data : TYPE
        Avalanche size list.
    xmin : TYPE
        Lower threshold for power-law estimation.
    xmax : TYPE
        Upper threshold for power-law estimation.
    alpha : TYPE
        Alpha parameter.
    titulo : TYPE
        Title for the plot.
    pdf : TYPE, optional
        DESCRIPTION. The default is 'prueba'. Title for the pdf file created.
    nb : TYPE, optional
        DESCRIPTION. The default is 10000. Number of bootstrap repetitions for empirical confidence intervals.

    Returns
    -------
    Creates a binned plot in scale log-log with the empirical distribution (and its uncertainities)
    and the estimated power-law distribution.

    '''
    data = data[(data >= xmin) & (data <= xmax)]

    # Log-log Histogram
    log_bins = np.histogram_bin_edges(np.log(data), bins='auto')  # Bins in log-scale
    bins = np.exp(log_bins)  # Back to original scale
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    # Averages for the bins (centers)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Bootstrap
    densb = np.zeros((nb, len(bin_mids))) #Inicialize

    for b in range(nb):
        sample_b = np.random.choice(data, size=len(data), replace=True)  # Resampling
        #sample_b = np.clip(sample_b, bins[0], bins[-1])  # Adjust values out of range
        hist_b, _ = np.histogram(sample_b, bins=bin_edges, density=True)
        densb[b, :] = hist_b

    # calculate confidence intervals 68% (standard deviation)
    ci_lower = np.percentile(densb, 16, axis=0)
    ci_upper = np.percentile(densb, 84, axis=0)
    
    # Power-law with alpha_MLE
    x_fit = np.linspace(xmin, xmax, 100)
    y_fit = pdf_power_law(x_fit, xmin, xmax, alpha)
    plt.figure(figsize=(10, 8))
    plt.plot(x_fit, y_fit, '--',linewidth=1.5, color=palette[1], label=f'P.L. ($\\alpha={alpha:.2f}$)', zorder=2)

    # Empirical distribution
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bar_heights = ci_upper - ci_lower
    plt.bar(bin_mids, ci_lower, width=bin_widths, color=palette[0], alpha=0.2, align='center', zorder=0
            ,edgecolor='black')
    plt.bar(bin_mids, bar_heights, width=bin_widths, bottom=ci_lower, align='center',
        alpha=0.35, color=palette[0], label="Data", edgecolor='black', zorder=0)

    # Confidence intervals
    plt.plot(bin_mids, hist, marker='o', color=palette[0], markersize=4, zorder=1)
    plt.hlines(hist, bin_edges[:-1], bin_edges[1:], color=palette[0], linewidth=2, zorder=1)

    # Labels and axis
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0, xmax)
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=30, length=15)
    ax.tick_params(axis='x', which='minor', labelsize=25, length=7)
    ax.tick_params(axis='y', which='major', labelsize=30, length=15)
    ax.tick_params(axis='y', which='minor', labelsize=25, length=7)
    ax.tick_params(axis='x', pad=10, length=10)
    ax.tick_params(axis='y', pad=10, length=10)
    
    plt.xlabel('Avalanche Size ($\mu$V)', fontsize=30, labelpad=12)
    plt.ylabel('Pdf', fontsize=30)
    #plt.title(f'Probability Density Function for Power Law in [${xmin:.2f}$, ${xmax:.0f}$] (Size) - {titulo}')
    legend = plt.legend(loc='best', fontsize=28, frameon=True, borderpad=0.5, framealpha=1, fancybox=False)
    legend.get_frame().set_linewidth(0.5)   
    plt.grid(True, linestyle='--',which='major', linewidth=0.5, alpha=0.7)
    
    plt.savefig(f'{pdf}.pdf') # saves the plot
    plt.show()

# %%% Survival function

def S_power_law(data, xmin, xmax, alpha, titulo, pdf='prueba'):
    '''

    Parameters
    ----------
    data : TYPE
        Avalcnhe size list.
    xmin : TYPE
        Lower threshold for power-law estimation.
    xmax : TYPE
        Upper threshold for power-law estimation.
    alpha : TYPE
        Alpha parameter.
    titulo : TYPE
        Title for the plot.
    pdf : TYPE, optional
        DESCRIPTION. The default is 'prueba'. Title for the pdf file created.

    Returns
    -------
    None.
    
    Creates a survival function plot for the sizes in the estimates range (truncated).
    Plots the estimated power-law survival distribution.

    '''
    data = data[(data >= xmin) & (data <= xmax)] # truncated data
    n = len(data) # Length of data
    # Empirical
    data_sorted = np.sort(data)
    S_empirical = 1-np.linspace(1/n, 1, n)
    
    # Estimated power law
    x_fit = np.linspace(xmin, xmax, 100)
    y_fit = ( (xmin / x_fit) ** (alpha - 1) - (xmin / xmax) ** (alpha - 1) ) / ( 1 - (xmin / xmax) ** (alpha - 1) )
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_fit, y_fit, '--', color=palette[1], label=f'P.L. ($\\alpha={alpha:.2f}$)', linewidth=1.5)
    plt.plot(data_sorted, S_empirical, 'o', linestyle='-', color=palette[0], markersize=1,  label="Data", zorder=2, linewidth=1.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Avalanche Size ($\mu$V)", fontsize=35, labelpad=10)
    plt.ylabel("$\\mathbb{P}(X)>x$", fontsize=35, labelpad=10)
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=30, length=15)
    ax.tick_params(axis='x', which='minor', labelsize=25, length=7)
    ax.tick_params(axis='y', which='major', labelsize=30, length=15)
    ax.tick_params(axis='y', which='minor', labelsize=25, length=7)
    ax.tick_params(axis='x', pad=10, length=10)
    ax.tick_params(axis='y', pad=10, length=10)

    #plt.title(f'Survival Function for Power Law in [${xmin:.2f}$, ${xmax:.0f}$] (Size) - {titulo}')
    plt.style.use('science')
    
    legend = plt.legend(loc='lower left', fontsize=28, frameon=True, borderpad=0.5, framealpha=1, fancybox=False)
    legend.get_frame().set_linewidth(0.5)
    plt.grid(True,which='major', linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    
    plt.savefig(f'{pdf}.pdf') # saves the plot
    plt.show()

# %%% Power-law fitting functions

def log_likelihood(alpha, data):
    '''

    Parameters
    ----------
    alpha : TYPE
        Alpha parameter.
    data : TYPE
        Avalanche sizes truncated list.

    Returns
    -------
    TYPE
        -Log Likelihood value.
    
    Calculates - log Likelihood for a certain alpha and range.

    '''
    N = len(data)
    a = np.min(data)
    b = np.max(data)
    r = a/b
    lng = N ** (-1) * np.sum(np.log(data))
    l = np.log((alpha-1)/(1-r**(alpha-1)))-alpha*(lng-np.log(a))-np.log(a)
    return -l  # Negativo porque queremos minimizar

def mle_alpha(data):
    '''

    Parameters
    ----------
    data : TYPE
        Avalanche size list.

    Returns
    -------
    alpha_mle : TYPE
        Alpha ML estimator.
        
    Calculates the optimal alpha value for a certain range of avalanches.

    '''
    initial_alpha = 1.2
    result = fmin(log_likelihood, initial_alpha, args=(data,), disp=False)
    alpha_mle = result[0]
    return alpha_mle

def montecarlo_power_law (a, r, N, alpha):
    '''

    Parameters
    ----------
    a : TYPE
        Lower threshold.
    r : TYPE
        r=a/b where a is the upper threshold.
    N : TYPE
        Length of data estimations.
    alpha : TYPE
        Alpha ML .

    Returns
    -------
    x : TYPE
        Synthetic data that follows a power law with alpha exponent.
    alpha_s : TYPE
        Alpha ML estimation from the synthetic data.
        
    Generates a sample of same length that follows a power law and estimates
    the optimal alpha exponent for that sample.

    '''
    u = np.random.rand(N)
    x = a/((1-(1-r**(alpha-1))*u)**(1/(alpha-1)))
    alpha_s = mle_alpha(x)
    return x, alpha_s

def ks_montecarlo (L, a, r, alpha, Lmont):
    '''
    
    Parameters
    ----------
    L : TYPE
        DESCRIPTION.
    a : TYPE
        Upper threshold.
    r : TYPE
        r=a/b where b is the lower threshold.
    alpha : TYPE
        Alpha ML estimation.
    Lmont : TYPE
        List of alpha values estimated from the synthetic data with
        Montecarlo simulations.

    Returns
    -------
    TYPE
        KS Test for the Montecarlo simulations.

    '''
        
    N = len(L)
    L = np.unique(L)
    Lmont = np.sort(Lmont)
    ns = np.sum(Lmont[:, None] >= L, axis=0)
    U = np.abs((1/(1-r**(alpha-1)))*((a/L)**(alpha-1)-r**(alpha-1))-ns/N)
    return np.max(U)

def ks_empirical (L, a, r, alpha):
    '''

    Parameters
    ----------
    L : TYPE
        DESCRIPTION.
    a : TYPE
        Upper threshold.
    r : TYPE
        r=a/b where b is the lower threshold..
    alpha : TYPE
        Alpha ML estimation.

    Returns
    -------
    TYPE
        KS Test for the estimated ML alpha (biased).

    '''
    N = len(L)
    L = np.unique(L)
    ne = np.arange(len(L),0,-1)
    U = np.abs((1/(1-r**(alpha-1)))*((a/L)**(alpha-1)-r**(alpha-1))-ne/N)
    return np.max(U)

def pvalor_plot(df, ysup, canal, ysupp=3, c=0.2):
    '''

    Parameters
    ----------
    df : TYPE
        Avalanches size.
    ysup : TYPE
        Upper limit for the y right axis in the plot.
    canal : TYPE
        'MR' or 'MOKE'. For title and pdf information.
    ysupp : TYPE, optional
        Upper limit for the y left axis in the plot (p-value).
        The default is 3.
    c : TYPE, optional
        Threshold for selecting a lower threshold and the corresponding alpha.
        The default is 0.2.
        An horizontal line is represented in the plot.

    Returns
    -------
    None.
    
    Creates a plot with the alpha ML values and p-values for different lower thresholds.
    It also represents confidence intervals for both, and horizontal lines at c and 0.05,
    to accept or reject the null hypothesis (p-value).

    '''
    pvalor_lista = df['pvalor']
    p_upper = df['pvalor']+df['u_pvalor']
    p_lower = df['pvalor']-df['u_pvalor']
    alpha_lista = df['alpha']
    a_upper = df['alpha']+df['u_alpha']
    a_lower = df['alpha']-df['u_alpha']
    rango = df['rango']
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # P-value axis
    ax1.set_ylim(0, ysupp)
    ax1.plot(rango, pvalor_lista, label="P-value", color=palette[0], linewidth=1)   
    ax1.axhline(y=0.05, color=palette[1], linestyle='--', label='P=0.05', zorder =0, linewidth=1.5)
    ax1.axhline(y= c, color=palette[3], linestyle='--', label= "P=0.20", zorder = 0, linewidth=1.5)
    ax1.fill_between(rango, p_upper, p_lower, alpha=0.4, color=palette[0])
    ax1.set_xlabel("$S_t$ ($\mu$V)", fontsize=35)
    ax1.set_ylabel("P-value", color=palette[0], fontsize=35)
    
    ax1.tick_params(axis='y', labelcolor=palette[0])
    ax1.tick_params(axis='both', which='major', labelsize=30, length=10)
    ax1.tick_params(axis='both', which='minor', labelsize=19, length=5)
    ax1.tick_params(axis='both', pad=10)
    
    # Alpha axis
    ax2 = ax1.twinx()  
    ax2.set_ylim(0, ysup)
    ax2.plot(rango, alpha_lista, label='$\\alpha_{MLE}$', color=palette[2], linewidth=1.5)
    ax2.fill_between(rango, a_upper, a_lower, alpha=0.4, color=palette[2])
    ax2.set_ylabel("$\\alpha_{MLE}$", color=palette[2], fontsize=35)
    ax2.tick_params(axis='y', labelcolor=palette[2])
    ax2.tick_params(axis='y', which='major', labelsize=30, length=10)
    ax2.tick_params(axis='y', which='minor', labelsize=19, length=5)
    ax2.tick_params(axis='y', pad=10)
    
    # Legend and title
    #plt.title(f"Power law fit with KS test for different lower thresholds - {canal}")
    lines = ax1.get_lines() + ax2.get_lines()  # Combina las líneas de ambos ejes
    labels = [line.get_label() for line in lines]  # Obtiene las etiquetas de todas las líneas
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray', 
               framealpha=0.9, fancybox=True, shadow=False, fontsize=23)
    fig.tight_layout()
    plt.savefig(f"pvalor_alfa_{canal}.pdf", dpi=300)
    plt.show()
    
def opt_alpha (data, k, b=25, ysup=3, ysupp=3, canal='MR', c=0.2, inc=0.01):
    '''

    Parameters
    ----------
    data : TYPE
        Avalanches size.
    k : TYPE
        Number of iterations in the Montecarlo Simulations.
    b : TYPE, optional
        Upper threshold. The default is 25.
    ysup : TYPE, optional
        DESCRIPTION. The default is 3.
    ysupp : TYPE, optional
        Upper limit for the y left axis in the plot (p-value).
        The default is 3.
    canal : TYPE, optional
        'MR' or 'MOKE'. For title and pdf information (plot).
         The default is 'MR'.
    c : TYPE, optional
        Threshold for selecting a lower threshold and the corresponding alpha (plot).
        The default is 0.2.
    inc : TYPE, optional
        Steps to increase the lower threshold. The default is 0.01.

    Returns
    -------
    df_opt : TYPE
        Dataframe with columns: p-value, u(p-value), alpha MLE, u(alpha MLE), minimum size, lower threshold.
    
    
    This function calculates alpha and the p-value for a selection of lower thresholds.
    It also represents the findings with a plot.
    '''
    data = data[data<= b]
    xmin = round(min(data),2)
    xmax = round(max(data),2)-0.02
    rango = np.arange (xmin, xmax, inc)
    pvalor_lista = []
    alpha_lista = []
    xmin_lista =[]
    ua_lista = []
    up_lista = []
    counter = 0
    for i in rango:
        data_i = data[data>=i]
        min_i = min(data_i)
        N = len(data_i)
        r = i/b
        alpha_i = mle_alpha(data_i)
        de = ks_empirical(data_i, i, r, alpha_i)
        alpha_m = []
        ds = []
        if counter % 10 == 0:  # Print every 10 steps
            print(f'{counter}/{len(rango)}')
            
        for j in range(k):
            data_mont, alpha_s = montecarlo_power_law (i, r, N, alpha_i)
            dsj = ks_montecarlo(data_i, i, r, alpha_s, data_mont)
            ds.append(dsj)
            alpha_m.append(alpha_s)
        # Alpha uncertainity with Montecarlo
        mean_alphas = np.mean(alpha_m)
        alpha_m = np.array(alpha_m)
        ua_i = np.sqrt(np.mean((alpha_m-mean_alphas)**2))
        ds = np.array(ds)
        p_value = np.sum(ds >= de) / k
        up_value = np.sqrt(p_value*(1-p_value)/k)
        
        pvalor_lista.append(p_value)
        up_lista.append(up_value)
        alpha_lista.append(alpha_i)
        xmin_lista.append(min_i)
        ua_lista.append(ua_i)
        counter +=1
    
    df_opt =  pd.DataFrame({"pvalor": pvalor_lista, "u_pvalor": up_lista, "alpha": alpha_lista, "u_alpha": ua_lista, "xmin": xmin_lista,"rango":rango })
    pvalor_plot(df_opt, ysup, canal, ysupp, c)
    
    return df_opt
# %%% Alpha & P-value test
# %%%% Dataset A
# %%%%% MR

dataMR_S = np.array(aval_MR['S'])
maxMR = 30
df_pvalorMR = opt_alpha(dataMR_S, k=500, b = maxMR, c=0.2, inc=0.01)
np.save("df_pvalorMR.npy", df_pvalorMR)
# %%%%% MOKE

dataMO_S = np.array(aval_MOKE['S'])
maxMO = 0.3
df_pvalorMO = opt_alpha(dataMO_S, k = 200, b = 0.3, ysup=4, canal='MOKE', c=0.2, inc=0.0002)
np.save("df_pvalorMO.npy", df_pvalorMO)
# %%%%% MOKE bivariate distribution

Lbi = np.load("Aval_biv_lista.npy")
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMO = aval_bi['MOKE']
maxMObi = 0.3

df_pvalorMO_bi = opt_alpha(aval_biMO, k=200, b = maxMObi, ysup=3, canal='MOKE', c=0.2, inc=0.0002)
np.save("df_pvalorMO_bi.npy", df_pvalorMO_bi)
# %%%%% MR bivariate distribution

Lbi = np.load("Aval_biv_lista.npy")
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMR = aval_bi['MR']
maxMRbi = 25

df_pvalorMR_bi = opt_alpha(aval_biMR, k=500, b = maxMRbi, ysup=3, canal='MR', c=0.2, inc=0.05)
np.save("df_pvalorMR_bi.npy", df_pvalorMR_bi)
# %%%% Dataset B
# %%%%% MR

dataMRB_S = np.array(aval_MR_B['S'])
maxMRB = 12
df_pvalorMRB = opt_alpha(dataMRB_S, k=300, ysup=12, ysupp=3, b = maxMRB, c=0.2, inc=0.05)
#np.save("df_pvalorMRB.npy", df_pvalorMRB)
# %%%%% MOKE

dataMOB_S = -np.array(aval_MOKE_B['S'])
maxMOB = 0.10
df_pvalorMOB = opt_alpha(dataMOB_S, k=100, b = maxMOB, ysup=5, ysupp=3, c=0.2, canal='MOKE', inc=0.0002)
#np.save("df_pvalorMOB.npy", df_pvalorMOB)

# %%%%% MR bivariate distribution

Lbi = np.load("Aval_biv_lista_B.npy")
aval_biB = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMRB = aval_biB['MR']
maxMRBbi = 14

df_pvalorMRB_bi = opt_alpha(aval_biMRB, k=200, b = maxMRBbi, ysup=10, canal='MR', c=0.2, inc=0.05)
#np.save("df_pvalorMRB_bi.npy", df_pvalorMRB_bi)
# %%%%% MOKE bivariate distribution
Lbi = np.load("Aval_biv_lista_B.npy")
aval_biB = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMOB = -aval_biB['MOKE']
maxMOBbi = 0.08

df_pvalorMOB_bi = opt_alpha(aval_biMOB, k=100, b = maxMOBbi, ysup=5, canal='MOKE', c=0.2, inc=0.0002)
#np.save("df_pvalorMOB_bi.npy", df_pvalorMOB_bi)
# %%%% Load files

columnas=["pvalor", "u_pvalor", "alpha", "u_alpha", "xmin", "rango"]
data = np.load("df_pvalorMR_bi.npy")
df_pvalorMR_bi = pd.DataFrame(data, columns=columnas)  
data = np.load("df_pvalorMO_bi.npy")
df_pvalorMO_bi = pd.DataFrame(data, columns=columnas)
data = np.load("df_pvalorMR.npy")
df_pvalorMR = pd.DataFrame(data, columns=columnas) 
data = np.load("df_pvalorMO.npy")
df_pvalorMO= pd.DataFrame(data, columns=columnas) 
data = np.load("df_pvalorMRB.npy")
df_pvalorMRB = pd.DataFrame(data, columns=columnas) 
data = np.load("df_pvalorMOB.npy")
df_pvalorMOB= pd.DataFrame(data, columns=columnas) 

dataMR_S = np.array(aval_MR['S'])
dataMO_S = np.array(aval_MOKE['S'])
dataMOB_S = -np.array(aval_MOKE_B['S'])
dataMRB_S = np.array(aval_MR_B['S'])

Lbi = np.load("Aval_biv_lista.npy")
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMO = aval_bi['MOKE']
aval_biMR = aval_bi['MR']

maxMRbi = 25
maxMObi = 0.3
maxMR = 30
maxMO = 0.3
maxMRB = 12
maxMOB = 0.1

# %%%% Plots
# %%%%% Dataset A bivariate distribution
pvalor_plot(df_pvalorMR_bi, 3, 'MR')
pvalor_plot(df_pvalorMO_bi, 2.5, 'MOKE')
# %%%%% Dataset A
pvalor_plot(df_pvalorMR, 5, 'MR')
pvalor_plot(df_pvalorMO, 3, 'MOKE')

# %%%%% Dataset B
pvalor_plot(df_pvalorMRB, 12, 'MR')
pvalor_plot(df_pvalorMOB, 5, 'MOKE')
# %%%% Optimal parameters
df_p1 = df_pvalorMR[df_pvalorMR['pvalor']>=0.2]
MRp_row = df_p1.iloc[0]
aMR = MRp_row.iloc[2]
minMR = MRp_row.iloc[5]
print(aMR)

df_p2 = df_pvalorMO[df_pvalorMO['pvalor']>=0.2]
MOp_row = df_p2.iloc[0]
aMO = MOp_row.iloc[2]
minMO = MOp_row.iloc[5]
print(aMO)


df_p3 = df_pvalorMO_bi[df_pvalorMO_bi['pvalor']>=0.2]
MO_bip_row = df_p3.iloc[0]
aMObi = MO_bip_row.iloc[2]
minMObi = MO_bip_row.iloc[5]
print(aMObi)

df_p4 = df_pvalorMR_bi[df_pvalorMR_bi['pvalor']>=0.2]
MR_bip_row = df_p4.iloc[0]
aMRbi = MR_bip_row.iloc[2]
minMRbi = MR_bip_row.iloc[5]
print(aMRbi)

df_p5 = df_pvalorMOB[df_pvalorMOB['pvalor']>=0.2]
MOBp_row = df_p5.iloc[0]
aMOB = MOBp_row.iloc[2]
minMOB = MOBp_row.iloc[5]
print(aMOB)

df_p6 = df_pvalorMRB[df_pvalorMRB['pvalor']>=0.2]
MRBp_row = df_p6.iloc[0]
aMRB = MRBp_row.iloc[2]
minMRB = MRBp_row.iloc[5]
print(aMRB)

# %%% Survival and Pdf Plots
# %%%% MOKE B

S_power_law(dataMOB_S, minMOB, maxMOB, aMOB, titulo='Canal MOKE', pdf='survival_ajuste_MOB')
plot_power_law(dataMOB_S, minMOB, maxMOB, aMOB, titulo='Canal MOKE', pdf='pdf_ajuste_MOB')
# %%%% MR B

S_power_law(dataMRB_S, minMRB, maxMRB, aMRB, titulo='Canal MR', pdf='survival_ajuste_MRB')
plot_power_law(dataMRB_S, minMRB, maxMRB, aMRB, titulo='Canal MR', pdf='pdf_ajuste_MRB')
# %%%% MOKE A bivariate

S_power_law(aval_biMO, minMObi,  maxMObi , aMObi , titulo='MOKE', pdf='survival_ajuste_MO_bi')
plot_power_law(aval_biMO, minMObi,  maxMObi, aMObi, titulo='MOKE', pdf='pdf_ajuste_MO_bi')
# %%%% MR A bivariate

S_power_law(aval_biMR, minMRbi,  maxMRbi , aMRbi , titulo='MR', pdf='survival_ajuste_MR_bi')
plot_power_law(aval_biMR, minMRbi,  maxMRbi, aMRbi, titulo='MR', pdf='pdf_ajuste_MR_bi')
# %%%% MR A

dataMR_S = aval_MR['S']
S_power_law(dataMR_S, minMR, maxMR, aMR, titulo='MR', pdf='survival_ajuste_MR')
plot_power_law(dataMR_S, minMR, maxMR, aMR, titulo='MR', pdf='pdf_ajuste_MR')
# %%%% MOKE A

dataMO_S = aval_MOKE['S']
S_power_law(dataMO_S, minMO, maxMO, aMO, titulo='Canal MOKE', pdf='survival_ajuste_MO')
plot_power_law(dataMO_S, minMO, maxMO, aMO, titulo='Canal MOKE', pdf='pdf_ajuste_MO')

# %% Bivariate distribution
# %%% Read Data
# %%%% Dataset A

MR = pd.read_csv('aval_MR_N.csv')
MO = pd.read_csv('aval_MOKE_N.csv')
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo

# %%%% Dataset B

MR = pd.read_csv('aval_MR_B_N.csv')
MO = pd.read_csv('aval_MOKE_B_N.csv')
rango=range(0,182)
# %%% Create Dataframe

MO = MO.rename(columns={'ti': 't'})
MO_bi = []
Lbi=[]

for i in rango:
    MO_aux = MO[MO['Archivo']==i]
    MR_aux = MR[MR['Archivo']==i]
    MO_aux = MO_aux.reset_index(drop=True)
    MR_aux = MR_aux.reset_index(drop=True)
    lmo = len(MO_aux)
    lmr = len(MR_aux)
    Li = pd.DataFrame([[0,0,0]], columns=['MO', 'MR', 't'])  # Fila vacía inicial
    for j in range(lmo):
        for k in range (lmr):
            #if  (MR_aux['ti'][k]+1 <= MO_aux['t'][j] <= MR_aux['tf'][k]+2):
            if MR_aux['ti'][k] <= MO_aux['t'][j] <= MR_aux['tf'][k]:
                new_row = pd.DataFrame([[MO_aux['S'][j], MR_aux['S'][k], k]], columns=['MO','MR','t'])
                Li = pd.concat([Li, new_row], ignore_index=True)                
                break
    Li = Li.iloc[1:].reset_index(drop=True)
    ti = np.unique(np.array(Li['t']))
    for l in ti:
        dfi = Li[Li['t']==l]
        MOi = dfi['MO'].sum()
        MRi = dfi['MR'].iloc[0]
        Lbi.append([MOi, MRi])
        
#np.save("Aval_biv_lista.npy", Lbi)
#np.save("Aval_biv_lista_B.npy", Lbi)              
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
#aval_bi['MOKE'] = -aval_bi['MOKE'] # For dataset B
# %%% Bivariate distribution plot

#Lbi = np.load("Aval_biv_lista_B.npy") # For Dataset B
Lbi = np.load('Aval_biv_lista.npy')
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
#aval_bi['MOKE'] = -aval_bi['MOKE'] # For dataset B
aval_bi2 = aval_bi # Scatter plot

# Filtered data for Dataset A only
aval_bi = aval_bi[(aval_bi['MOKE']>= minMObi) & (aval_bi['MOKE']<= maxMObi)].reset_index(drop=True)
aval_bi = aval_bi[(aval_bi['MR']>= minMRbi) & (aval_bi['MR']<= maxMRbi)].reset_index(drop=True)

X = np.log(aval_bi['MOKE']).values
Y = np.log(aval_bi['MR']).values
Xp = np.log(aval_bi['MOKE']).values.reshape(-1, 1)
Xp = sm.add_constant(Xp)

# New lists for the fit MOKE vs MR

Yp = np.log(aval_bi['MR']).values.reshape(-1, 1)
Yp = sm.add_constant(Yp)


# MR vs MOKE fit

model1 = sm.OLS(Y,Xp)
results1 = model1.fit()
Y_pred = results1.predict(Xp)
b1, m1 = results1.params
ub1, um1 = results1.bse
uY = abs(Y_pred) * np.sqrt((X * um1) ** 2 + ub1 ** 2)
upper_bound1 = np.exp(Y_pred) + uY
lower_bound1 = np.exp(Y_pred) - uY

# MOKE vs MR fit

model2 = sm.OLS(X,Yp)
results2 = model2.fit()
X_pred = results2.predict(Yp)
b2, m2 = results2.params
ub2, um2 = results2.bse
uX = abs(X_pred) * np.sqrt((Y * um2) ** 2 + ub2 ** 2)
upper_bound2 = np.exp(X_pred) + uX
lower_bound2 = np.exp(X_pred) - uX

# Plot

plt.figure(figsize=(14,8))
palette = sns.color_palette("muted")

sns.scatterplot(data=aval_bi2, x='MOKE', y='MR', label='Exp. data', alpha=0.3, color=palette[0]) 
sns.scatterplot(data=aval_bi, x='MOKE', y='MR', label='Filtered data', alpha=0.3, color=palette[4]) # Only for Dataset A
sns.lineplot(x=aval_bi['MOKE'], y=np.exp(Y_pred), label='MR $\mid$ MOKE', color=palette[1], linewidth=1.5)
sns.lineplot(y=aval_bi['MR'], x=np.exp(X_pred), label='MOKE $\mid$ MR', color=palette[2], linewidth=1.5)

sorted_indices1 = np.argsort(aval_bi['MOKE'])
sorted_indices2 = np.argsort(aval_bi['MR'])

# Ordered dataset to use fill_between

x_sorted = aval_bi['MOKE'][sorted_indices1]
y_lower_sorted = lower_bound1[sorted_indices1]
y_upper_sorted = upper_bound1[sorted_indices1]
plt.fill_between(x_sorted, y_upper_sorted, y_lower_sorted, alpha=0.6, color=palette[1])

y_sorted = aval_bi['MR'][sorted_indices2]
x_lower_sorted = lower_bound2[sorted_indices2]
x_upper_sorted = upper_bound2[sorted_indices2]
#plt.fill_betweenx(y_sorted, x_upper_sorted, x_lower_sorted, alpha=0.6, color=palette[2])

# Average fit

m3 = (m1+1/m2)/2
b3 = (b1-b2/m2)/2
Y_pred2 = X*m3+b3
sns.lineplot(x=aval_bi['MOKE'], y=np.exp(Y_pred2), label='Average fit', color=palette[3], linewidth=1.5)

um3 = np.sqrt(um1**2+(um2/(m2**2))**2)/2
ub3 =  np.sqrt(ub1**2+(ub2/m2)**2+(b2*um2/(m2**2))**2)/2
print(f"{b1:.3f} {ub1:.3f} {m1:.3f} {um1:.3f}")
print(f"{b2:.3f} {ub2:.3f} {m2:.3f} {um2:.3f}")
print(f"{b3:.3f} {ub3:.3f} {m3:.3f} {um3:.3f}")
r2_1 = results1.rsquared
print(f"R² ajuste 1 (MR vs MOKE): {r2_1:.3f}")
r2_2 = results2.rsquared
print(f"R² ajuste 2 (MOKE vs MR): {r2_2:.3f}")

residuals = Y - Y_pred2
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y - np.mean(Y))**2)
r2_3 = 1 - ss_res/ss_tot
print(f"R² ajuste 3 (promedio): {r2_3:.3f}")

uY3 = abs(Y_pred) * np.sqrt((X * um3) ** 2 + ub3 ** 2)
upper_bound3 = np.exp(Y_pred) + uY3
lower_bound3 = np.exp(Y_pred) - uY3
y_lower_sorted3 = lower_bound3[sorted_indices1]
y_upper_sorted3 = upper_bound3[sorted_indices1]
plt.fill_between(x_sorted, y_upper_sorted3, y_lower_sorted3, alpha=0.7, color=palette[3])


# Labels and axis

#plt.ylim(0.1,)
plt.xlim(0.004,)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('MR Avalanche Size ($\mu$V)', fontsize=23, labelpad=10)
plt.xlabel('MOKE Avalanche Size', fontsize=23)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='x', which='major', length=10)
plt.tick_params(axis='x', which='minor', length=3)
plt.tick_params(axis='y', which='major', length=10)
plt.tick_params(axis='y', which='minor', length=3) 

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder =0)

# Legend

plt.legend(loc='lower right', fontsize=18, frameon=True, facecolor='white', 
           edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False)

# Save and show

plt.savefig('bivariada_dist_N.png')
plt.show()