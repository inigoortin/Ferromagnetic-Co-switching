# %% Librerías
import pandas as pd
import numpy as np
import seaborn as sns
#from scipy.stats import powerlaw
from collections import Counter
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#from lifelines import KaplanMeierFitter
#from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy.optimize import fmin

import random
import matplotlib.pyplot as plt
import scienceplots # gráficos con estilo unificado y formato latex
plt.style.use('science')
palette = sns.color_palette("muted")

# %% Preparación archivos avalanchas MR y MOKE

columnas = ['Corriente', 'MOKE', 'MR']
aval_MR = pd.DataFrame(columns=['MR', 'Corriente', 'dMR', 'ti', 'tf', 'T', 'S', 'Archivo'])
aval_MOKE = pd.DataFrame(columns=['MOKE', 'Corriente', 'dMOKE', 't', 'S', 'Archivo'])
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#rango = range(0,181) # Archivo 165 es erróneo

for i in rango:
    url = f'Datos/hysteresis_deg_{i}.dat'    
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
#aval_MR.to_csv('aval_MR.csv', index=False)
#aval_MOKE.to_csv('aval_MOKE.csv', index=False)
# %% Incluye avalanchas MR T=1
# También se estandariza MOKE teniendo en cuenta que es posible que T>1

columnas = ['Corriente', 'MOKE', 'MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo'])
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#rango = range(0,181)

for i in rango:
    url = f'Datos/hysteresis_deg_{i}.dat'    
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

# %% 
columnas = ['Corriente', 'MOKE', 'MR']
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
inicio_moke = []
for i in rango:
    url = f'Datos/hysteresis_deg_{i}.dat'    
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df = df[df['Corriente']<-0.026]
    inicio_moke.append(min(df['MOKE']))
m0 = np.mean(inicio_moke)
std_m0 = np.std(inicio_moke)
# %% Normalización MOKE Datos A
# También se estandariza MOKE teniendo en cuenta que es posible que T>1

columnas = ['Corriente', 'MOKE', 'MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo', 'Corriente'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo', 'Corriente'])
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#rango = range(0,181)

for i in rango:
    url = f'Datos/hysteresis_deg_{i}.dat'    
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
# %% Preparación archivos avalanchas datos B

columnas = ['Corriente', 'MOKE', 'Hall','MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo'])
rango = range(0,182)
for i in rango:
    url = f'Datos3/hysteresis_deg_{i}.dat'    
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

# aval_MR_B contiene los datos obtenidos con umbral fijo (10000)

# %% Avalanchas datos B normalizadas
columnas = ['Corriente', 'MOKE', 'Hall','MR']
aval_MR1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti', 'tf', 'T', 'S', 'Archivo', 'Corriente'])
aval_MOKE1 = pd.DataFrame([[0,0,0,0,0,0]], columns=['ti','tf','T', 'S', 'Archivo','Corriente'])
rango = range(0,182)
for i in rango:
    url = f'Datos3/hysteresis_deg_{i}.dat'    
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

# %% Generar predictores
# Obtenemos las predicciones de avalanchas con el método
# En un primer cribaje obtenemos los precursores de los precursores
# Posteriormente, guardamos únicamente el último elemento de cada "cluster"

# De momento funciona para datos viejos y nuevos, hay que observar los datos viejos por si hace falta hacer un nuevo ajuste

def crear_pred (C1=0.8, C2=0.2, p=0, datos='viejo'):
    if datos == 'viejo':
        rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
        columnas = ['Corriente', 'MOKE', 'MR']
    if datos == 'nuevo':
        rango = range(0,98)
        columnas = ['Corriente', 'MOKE', 'Hall','MR']
    pred = [] # Lista de 2-tuplas donde se guarda el tiempo y archivo
    for j in rango:
        predj = []
        if datos == 'viejo':
            url = f'Datos/hysteresis_deg_{j}.dat'
        if datos == 'nuevo':
            url = f'Datos_nuevos/hysteresis_deg_{j}.dat'    
        df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
        df['t']=range(len(df))
        start = df['MR'][299]
        for i in range(300, len(df['t'])):
            df_aux = df[(df['t'] >= i - 10) & (df['t'] <= i)]
            df_aux = df_aux[['MR', 't']]
            x = df_aux['MR'][i]
            x_min = min(start, x)
            x_max = df_aux['MR'].max()
            if (x_max != x_min): paval = (x-x_min)/(x_max-x_min)
            else: paval = 1
            paval2 = x_max - x
            if p== 0:
                if (paval < C1) & (paval2 > C2):
                    predj.append([i,j])
            elif p == 1:
                if paval < C1:
                    predj.append([i,j])
            elif p == 2:
                if paval2 > C2:
                    predj.append([i,j])
        l = len(predj) - 1
        k = 1 # índice secundario para pasar de un grupo de predictores a otro
        while l >= 1:
            if k ==1: # Guardamos el final
                final = predj[l]
            if predj[l][0] == predj[l-1][0] + 1:
                predj.pop(l)
                k = 2 # Recorremos el índice temporal hasta encontrar el inicio
            else:
                pred.append(final)
                k = 1 # Reinstauramos el índice para marcar un nuevo final
            l -= 1       
    df_pred = pd.DataFrame(pred, columns=['t','Archivo'])
    return df_pred
    
    #df_pred.to_csv('df_pred.csv', index=False)

# %% LTM vs STM method - DESCONTINUADO

def LTA_vs_STA (short_w = 5, large_w = 25, R0=0.5):
    #rango = range(0,98)
    rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
    #columnas = ['Corriente', 'MOKE', 'Hall','MR']
    columnas = ['Corriente', 'MOKE', 'MR']
    pred = [] # Lista de 2-tuplas donde se guarda el tiempo y archivo
    for j in rango:
        predj = []
        url = f'Datos/hysteresis_deg_{j}.dat'    
        #url = f'Datos_nuevos/hysteresis_deg_{j}.dat'    
        df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
        df['t']=range(len(df))
        for i in range(300, len(df['t'])):
            df_LTA = df[(df['t'] >= i - large_w) & (df['t'] <= i)].reset_index()
            df_STA = df[(df['t'] >= i - short_w) & (df['t'] <= i)].reset_index()
            df_LTA = df_LTA['MR']
            df_STA = df_STA['MR']
            LTA = np.mean(df_LTA)
            STA = np.mean(df_STA)
            R = STA/LTA
            #LTA = np.mean(np.diff(df_LTA.to_numpy()))
            #STA = np.mean(np.diff(df_STA.to_numpy()))
            if R<R0:
                predj.append([i,j])
        l = len(predj) - 1
        k = 1 # índice secundario para pasar de un grupo de predictores a otro
        while l >= 1:
            if k ==1: # Guardamos el final
                final = predj[l]
            if predj[l][0] == predj[l-1][0] + 1:
                predj.pop(l)
                k = 2 # Recorremos el índice temporal hasta encontrar el inicio
            else:
                pred.append(final)
                k = 1 # Reinstauramos el índice para marcar un nuevo final
            l -= 1       
    df_pred = pd.DataFrame(pred, columns=['t','Archivo'])
    return df_pred

# %% Funciones para verificar el acierto de una lista de predictores
# La segunda función crea y verifica predictores aleatorios a partir de una distribución normal

def verificar_prediccion (df, df_aval, ventana=4):
    l = len(df)
    df['Result']=0
    df_aval['M-Aux'] = 0
    for i in range(l):
        t = df['t'][i]
        file = df['Archivo'][i]
        df_aux = df_aval[df_aval['Archivo']== file]
        ti = df_aux['ti'].to_list()
        #tf = df_aux['tf'].to_list()

        #for taval, tfaval in zip(ti, tf):  # Recorre cada valor de la lista de avalanchas para ese archivo
         #   if (t + ventana >= taval) and (t <= tfaval):  # Verificas si ti_aval está en el rango [t+1, t+ventana]
        for taval in ti:  # Recorre cada valor de la lista de avalanchas para ese archivo
            if t  <= taval <= t + ventana: 
                df.at[i, 'Result'] = 1
                df_aval.loc[(df_aval['ti'] == taval) & (df_aval['Archivo'] == file), 'M-Aux'] = 1
                #break # Aumentaba erróneamente el número de falsos positivos para el método M1
    A = df_aval['M-Aux'].sum()
    FP = l - df['Result'].sum()
    TE = 1-FP/l #TE: Tasa de éxito del método
    return ([A,FP, TE, l])

def metodo_nulo (k, media, sd, metodo, df_aval, ventana=4):
    A_lista = []
    FP_lista = []
    TE_lista = []
    for i in range(k):
        num_pred = np.random.normal(loc=media, scale=sd, size=201)
        num_pred = np.round(num_pred).astype(int)
        num_pred = np.maximum(num_pred, 0)
        rango_min, rango_max = 300, 1040-ventana
        resultado = [random.sample(range(rango_min, rango_max + 1), num) for num in num_pred]
        df_i = pd.DataFrame(
        [(valor, i if i < 165 else i + 1) for i, sublista in enumerate(resultado) for valor in sublista],
        columns=["t", "Archivo"])
        L = verificar_prediccion(df_i, df_aval)
        A_lista.append(L[0])
        FP_lista.append(L[1])
        TE_lista.append(L[2])
        if i % 50 == 0:
            print(f"Procesados: {i}")
    return A_lista, FP_lista, TE_lista

# %% Cargar archivos
url1 ='df_pred.csv'
url2 ='aval_MR_N.csv'
url3 = 'aval_MOKE_N.csv'

df_pred = pd.read_csv(url1)
aval_MR = pd.read_csv(url2)
aval_MOKE = pd.read_csv(url3)

# %% Archivos datos B
aval_MR_B = pd.read_csv('aval_MR_B_N.csv')
aval_MOKE_B = pd.read_csv('aval_MOKE_B_N.csv')

# %% Explicación método
# Paso 1: Calculamos la distribución del número de avalanchas de cada serie temporal
# Paso 2: Determinar el número de precursores de cada serie
# Paso 3: Generar los precursores aleatoriamente
# Paso 4: Verificar 


# Todas las listas tienen 1040 tiempos
#aval_long = []
#rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#for i in rango:
#    url = f'Datos/hysteresis_deg_{j}.dat'    
#    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
#    l = len(df)
#    aval_long.append(l)

# Hemos generalizado este proceso permitiendo que el proceso nulo siga una distribución concreta del método M1
# Para obtener resultados fiables se repite el procedimiento k veces (habitualmente por tiempo de computación 100)

# Generalizamos el método con dos funciones
# Ganamos flexibilidad para la distribución del número de predictores por serie
# Obtenemos estadísticas de las repeticiones

# %% Variación C1 y C2 simultánea

rangoi = np.arange(0.75, 1.03, 0.05)
rangoj = np.arange(0.15, 0.45, 0.05)
matrix_TE = np.empty((len(rangoi), len(rangoj)))
matrix_A = np.empty((len(rangoi), len(rangoj)))
k=1
idx_i = 0
for i in rangoi:
    idx_j = 0
    for j in rangoj:
        print(k)
        dfaux = crear_pred(C1=i, C2=j)
        L = verificar_prediccion(dfaux, aval_MR1)
        matrix_TE[idx_i][idx_j] = L[2]
        matrix_A[idx_i][idx_j] = L[0]
        idx_j += 1 
        k+=1
    idx_i += 1
# %% Variación C1 y C2 simultánea mejor zona (zoom)

rangoiz = np.linspace(0.92, 1.00, num=9)
rangojz = np.arange(0.28, 0.42, 0.02)
matrix_TEz = np.empty((len(rangoiz), len(rangojz)))
matrix_Az = np.empty((len(rangoiz), len(rangojz)))
k=1
idx_i = 0
for i in rangoiz:
    idx_j = 0
    for j in rangojz:
        print(k)
        dfaux = crear_pred(C1=i, C2=j)
        L = verificar_prediccion(dfaux, aval_MR1)
        matrix_TEz[idx_i][idx_j] = L[2]
        matrix_Az[idx_i][idx_j] = L[0]
        idx_j += 1 
        k+=1
    idx_i += 1
# %% Guardar y cargar matrices
#np.save("matriz_AV.npy", matrix_A)
#np.save("matriz_AVz.npy", matrix_Az)
#np.save("matriz_TE.npy", matrix_TE)
#np.save("matriz_TEz.npy", matrix_TEz)

matrix_A = np.load("matriz_AV.npy")
matrix_Az = np.load("matriz_AVz.npy")
matrix_TE = np.load("matriz_TE.npy")
matrix_TEz = np.load("matriz_TEz.npy")

# %% Función gráficos Heatmap

def heatmap (matrix, rangoi, rangoj, clase = 0):
    fig, ax = plt.subplots(figsize=(8,6))

    # Usamos imshow para crear el mapa de calor
    cax = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(rangoj)))  # Establecer las posiciones de las etiquetas en el eje X
    ax.set_xticklabels([f"{x:.2f}" for x in rangoj])  # Formatear las etiquetas con dos decimales

    ax.set_yticks(np.arange(len(rangoi)))  # Establecer las posiciones de las etiquetas en el eje Y
    ax.set_yticklabels([f"{y:.2f}" for y in rangoi])  # Formatear las etiquetas con dos decimales

    # Agregar un colorbar para mostrar la escala de colores
    fig.colorbar(cax)

    # Etiquetas y título
    ax.set_xlabel('$x_{max}-x$', fontsize=14)
    ax.set_ylabel('$\\frac{x-x_{min}}{x_{max}-x_{min}}$',fontsize=14, rotation=0, labelpad=25)
    plt.xticks(rotation=90)   # Rotar las etiquetas del eje X si es necesario
    if clase == 0:
        ax.set_title('Mapa de Calor de TE', fontsize=14)
        plt.savefig('Calor_TE.pdf')
    elif clase == 1:
        ax.set_title('Mapa de Calor de Avalanchas', fontsize=14)
        plt.savefig('Calor_AV.pdf')
    # Mostrar el gráfico
    plt.show()
    
# %% Heatmaps
# Heatmap de TE
heatmap(matrix_TE, rangoi, rangoj, clase=0)

# Heatmap de Avalanchas
heatmap(matrix_A, rangoi, rangoj, clase=1)

# Heatmap de TE zoom
heatmap(matrix_TEz, rangoiz, rangojz, clase=0)

# Heatmap de Avalanchas zoom
heatmap(matrix_Az, rangoiz, rangojz, clase=1)

# %% Función gráfico variación parámetros
def plot_param_M1 (x, y, clase = 0, C = 1):
    plt.figure(figsize=(8,6))
    plt.plot(x, y, color=palette[0])
    if C == 1:
        plt.xlabel('$\\frac{x-x_{min}}{x_{max}-x_{min}}$')
        plt.axvline(x=0.98, color=palette[1], linestyle='--', linewidth=1, label='$\\frac{x-x_{min}}{x_{max}-x_{min}}= 0.98$')
        plt.legend()
    elif C==2: 
        plt.xlabel('$x_{max}-x$')
    if clase == 0:
        plt.ylabel('$1-\\frac{FP}{N}$')
        #plt.savefig(f'TE_C{C}.pdf')
    elif clase == 1: 
        plt.ylabel('Avalanchas detectadas')
        #plt.savefig(f'Avalanchas_C{C}.pdf')
    plt.show()
# %% Variación parámetro C2 (xmax-x)
rango = np.arange(0.10, 0.51, 0.01)
LM2 = []
AV2 = []
k=1
for i in rango:
    print(k)
    dfaux = crear_pred(C1=0.8, C2=i, p=2)
    L = verificar_prediccion(dfaux, aval_MR1)
    LM2.append(L[2])
    AV2.append(L[0])
    k+=1

# %% Gráficos parámetro C2

#np.save("LM2.npy", LM2)
#np.save("AV2.npy", AV2)
LM2 = np.load("LM2.npy")
AV2 = np.load("AV2.npy")
rango = np.arange(0.10, 0.51, 0.01)
plot_param_M1(rango, LM2, clase = 0, C =2)
plot_param_M1(rango, AV2, clase = 1, C =2)

# %% Variación parámetro C1 (relativo a xmin)
rango = np.arange(0.85, 1.0, 0.01)
LM1 = []
AV1 = []
k = 1
for i in rango:
    print(k)
    dfaux = crear_pred(C1=i, C2=0.8, p=1)
    L = verificar_prediccion(dfaux, aval_MR1)
    LM1.append(L[2])
    AV1.append(L[0])
    k+=1
print(LM1)
# %% Gráficos parámetro C1
# np.save("LM1.npy", LM1)
# np.save("AV1.npy", AV1)
LM1 = np.load("LM1.npy")
AV1 = np.load("AV1.npy")
rango = np.arange(0.85, 1.0, 0.01)
#rango = np.concatenate([np.arange(0.80, 1.0, 0.01), [0.995, 1.00]])
plot_param_M1(rango, LM1, clase = 0, C =1)
plot_param_M1(rango, AV1, clase = 1, C =1)
# %% Simulaciones método nulo
# %% Simulaciones método nulo
# Predictores siguen distribución del método 1

df_pred = crear_pred(C1=0.98, C2=0.8, p=1)
df_pred_Archivo = df_pred['Archivo'].to_list()
contador2 = Counter(df_pred_Archivo)
num_aval2 = list(contador2.values())
media2 =  np.mean(num_aval2)
sd2 = np.std(num_aval2, ddof=0) # ddof=0 es desviación estándar poblacional
A_nulo, FP_nulo, TE_nulo = metodo_nulo (10000, media2, sd2, "nulo", aval_MR, ventana=4)
print(np.mean(np.array(TE_nulo)))
#np.save("FP_nulo_MR_prueba.npy", FP_nulo)
#np.save("A_nulo_MR_prueba.npy", A_nulo)
#np.save("TE_nulo_MR_prueba.npy", TE_nulo)
# %% Guardar y cargar simulaciones del método nulo
#np.save("FP_nulo_MR1.npy", FP_nulo)
#np.save("A_nulo_MR1.npy", A_nulo)
#np.save("TE_nulo_MR1.npy", TE_nulo)

FP_nulo = np.load("FP_nulo_MR1.npy")
A_nulo = np.load("A_nulo_MR1.npy")
TE_nulo = np.load("TE_nulo_MR1.npy")

#FP_nulo = np.load("FP_nulo_MR_prueba.npy")
#A_nulo = np.load("A_nulo_MR_prueba.npy")
#TE_nulo = np.load("TE_nulo_MR_prueba.npy")
# %% Predicción óptima
df_pred = crear_pred(C1=0.98, C2=0.8, p=1)
L_M1 = verificar_prediccion(df_pred, aval_MR1)
AV1 = L_M1[0]
FP1 = L_M1[1]
TE1 = L_M1[2]
print(L_M1)

# %% Método nulo: Distribución tasa de acierto

# Hist con función de densidad suavizada
# Estimación de densidad con KDE
kde = gaussian_kde(TE_nulo)
x_vals = np.linspace(min(TE_nulo), max(TE_nulo), 1000)  # Rango de valores para evaluar la densidad
y_vals = kde(x_vals)
mean_val = np.mean(TE_nulo)

# Calcular cuantiles
q_low, q_high = np.percentile(TE_nulo, [2.5, 97.5])

# Graficar histograma con diferentes transparencias
plt.figure(figsize=(8, 6), dpi=300)
n, bins, patches = plt.hist(TE_nulo, bins=30, density=True, alpha=0.6, color=palette[0], label='$f(x)$')

# Ajustar la transparencia de cada barra
for patch, bin_edge in zip(patches, bins[:-1]):  
    if q_low <= bin_edge <= q_high:
        patch.set_alpha(0.35)  # Más opaco dentro del rango
    else:
        patch.set_alpha(0.2)  # Más transparente fuera del rango

# Graficar la KDE
plt.plot(x_vals, y_vals, color=palette[0], label="KDE", linewidth=2)

# Graficar la media
plt.axvline(x=mean_val, color=palette[1], linestyle='-', linewidth=1.5, label=f'Mean = {mean_val:.3f}')
#plt.axvline(x=TE1, color=palette[1], linestyle='--', label=f'SR = {TE1:.2f}')

plt.xlabel("Success rate ($x$)")
plt.ylabel("$f(x)$")
plt.title("Probability density function - Success rate")
plt.legend()
plt.savefig('Nulo_TE_Hist_MR_prueba.pdf')
plt.show()

# %% Método nulo: Distribución avalanchas

# Estimación de densidad con KDE
kde = gaussian_kde(A_nulo)
x_vals = np.linspace(min(A_nulo), max(A_nulo), 1000)  # Rango de valores para evaluar la densidad
y_vals = kde(x_vals)
mean_val = np.mean(A_nulo)

# Calcular cuantiles
q_low, q_high = np.percentile(A_nulo, [2.5, 97.5])

# Graficar histograma con diferentes transparencias
plt.figure(figsize=(8, 6), dpi=300)
n, bins, patches = plt.hist(A_nulo, bins=30, density=True, alpha=0.6, color=palette[0], label='$f(x)$')

# Ajustar la transparencia de cada barra
for patch, bin_edge in zip(patches, bins[:-1]):  
    if q_low <= bin_edge <= q_high:
        patch.set_alpha(0.35)  # Más opaco dentro del rango
    else:
        patch.set_alpha(0.2)  # Más transparente fuera del rango

# Graficar la KDE
plt.plot(x_vals, y_vals, color=palette[0], label="KDE", linewidth=2)

# Graficar la media
plt.axvline(x=mean_val, color=palette[1], linestyle='-', linewidth=1.5, label=f'Mean = {mean_val:.0f}')
#plt.axvline(x=AV1, color=palette[1], linestyle='--', label=f'Avalanches = {AV1:.0f}')

plt.xlabel("Avalanches")
plt.ylabel("$f(x)$")
plt.title("Probability density function - Avalanches")
plt.legend(loc='best')
plt.savefig('Nulo_AV_Hist_MR_prueba.pdf')
plt.show()

# %% Comparamos resultados con los nuevos datos
df_predN = crear_pred(C1=0.98, C2=0.2, p=1, datos='nuevo')
verificar_prediccion(df_predN, aval_MRN)

# %% Probability Density Function
# Esta función ajusta y grafica una pdf de ley de potencias truncada.

def pdf_power_law (data, a, b, alpha):
    y = (alpha-1)/(a**(1-alpha)-1/(b**(alpha-1)))*((1/data)**alpha)
    return y

def plot_power_law(data, xmin, xmax, alpha, titulo, pdf='prueba', nb=10000):
    data = data[(data >= xmin) & (data <= xmax)]

    # Histograma en escala logarítmica
    log_bins = np.histogram_bin_edges(np.log(data), bins='auto')  # Bins en escala log
    bins = np.exp(log_bins)  # Volver a escala original
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    # Calcular medias de los bins (centros)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Bootstrap para estimar intervalos de confianza
    densb = np.zeros((nb, len(bin_mids))) # Inicializa, crea una matriz de ceros

    for b in range(nb):
        sample_b = np.random.choice(data, size=len(data), replace=True)  # Resampling con reemplazo
        #sample_b = np.clip(sample_b, bins[0], bins[-1])  # Ajuste de valores fuera del rango
        hist_b, _ = np.histogram(sample_b, bins=bin_edges, density=True)
        densb[b, :] = hist_b

    # Calcular intervalos de confianza del 68%
    ci_lower = np.percentile(densb, 16, axis=0)
    ci_upper = np.percentile(densb, 84, axis=0)
    
    # Graficar la ley de potencias ajustada con alpha_MLE
    x_fit = np.linspace(xmin, xmax, 100)
    y_fit = pdf_power_law(x_fit, xmin, xmax, alpha)
    plt.figure(figsize=(10, 8))
    plt.plot(x_fit, y_fit, '--',linewidth=1.5, color=palette[1], label=f'P.L. ($\\alpha={alpha:.2f}$)', zorder=2)

    # Graficar la distribución empírica con incertidumbre bootstrap
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bar_heights = ci_upper - ci_lower
    plt.bar(bin_mids, ci_lower, width=bin_widths, color=palette[0], alpha=0.2, align='center', zorder=0
            ,edgecolor='black')
    plt.bar(bin_mids, bar_heights, width=bin_widths, bottom=ci_lower, align='center',
        alpha=0.35, color=palette[0], label="Data", edgecolor='black', zorder=0)

    # Agregar barras de error con los intervalos de confianza
    plt.plot(bin_mids, hist, marker='o', color=palette[0], markersize=4, zorder=1)
    plt.hlines(hist, bin_edges[:-1], bin_edges[1:], color=palette[0], linewidth=2, zorder=1)

    # Configuración de ejes y etiquetas
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
    
    plt.xlabel('Avalanche Size', fontsize=30, labelpad=12)
    plt.ylabel('Pdf', fontsize=30)
    #plt.title(f'Probability Density Function for Power Law in [${xmin:.2f}$, ${xmax:.0f}$] (Size) - {titulo}')
    legend = plt.legend(loc='best', fontsize=28, frameon=True, borderpad=0.5, framealpha=1, fancybox=False)
    legend.get_frame().set_linewidth(0.5)   
    plt.grid(True, linestyle='--',which='major', linewidth=0.5, alpha=0.7)
    
    plt.savefig(f'{pdf}.pdf') # para comprobar como quedan los gráficos
    plt.show()

# %% Survival

# Esta función ajusta y grafica una función de supervivencia para una ley de potencias truncada.
def S_power_law(data, xmin, xmax, alpha, titulo, pdf='prueba'):
    data = data[(data >= xmin) & (data <= xmax)]
    n = len(data)      # Número de datos

    # Empírica
    data_sorted = np.sort(data)
    S_empirical = 1-np.linspace(1/n, 1, n)
    
    # Teórica
    x_fit = np.linspace(xmin, xmax, 100)
    y_fit = ( (xmin / x_fit) ** (alpha - 1) - (xmin / xmax) ** (alpha - 1) ) / ( 1 - (xmin / xmax) ** (alpha - 1) )
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_fit, y_fit, '--', color=palette[1], label=f'P.L. ($\\alpha={alpha:.2f}$)', linewidth=1.5)
    plt.plot(data_sorted, S_empirical, 'o', linestyle='-', color=palette[0], markersize=1,  label="Data", zorder=2, linewidth=1.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Avalanche Size", fontsize=35, labelpad=10)
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
    
    plt.savefig(f'{pdf}.pdf') # para comprobar como quedan los gráficos
    plt.show()

# %% Funciones Elección alpha: Power Law

def log_likelihood(alpha, data):
    N = len(data)
    a = np.min(data)
    b = np.max(data)
    r = a/b
    lng = N ** (-1) * np.sum(np.log(data))
    l = np.log((alpha-1)/(1-r**(alpha-1)))-alpha*(lng-np.log(a))-np.log(a)
    return -l  # Negativo porque queremos minimizar

# Función para calcular el valor de alpha utilizando MLE
def mle_alpha(data):
    initial_alpha = 1.2
    result = fmin(log_likelihood, initial_alpha, args=(data,), disp=False)
    alpha_mle = result[0]
    return alpha_mle

def montecarlo_power_law (a, r, N, alpha):
    u = np.random.rand(N)
    x = a/((1-(1-r**(alpha-1))*u)**(1/(alpha-1)))
    alpha_s = mle_alpha(x)
    return x, alpha_s

def ks_montecarlo (L, a, r, alpha, Lmont):
    N = len(L)
    L = np.unique(L)
    Lmont = np.sort(Lmont)
    ns = np.sum(Lmont[:, None] >= L, axis=0)
    U = np.abs((1/(1-r**(alpha-1)))*((a/L)**(alpha-1)-r**(alpha-1))-ns/N)
    return np.max(U)

def ks_empirical (L, a, r, alpha):
    N = len(L)
    L = np.unique(L)
    ne = np.arange(len(L),0,-1)
    U = np.abs((1/(1-r**(alpha-1)))*((a/L)**(alpha-1)-r**(alpha-1))-ne/N)
    return np.max(U)

def pvalor_plot(df, ysup, canal, ysupp=3, c=0.2):
    pvalor_lista = df['pvalor']
    p_upper = df['pvalor']+df['u_pvalor']
    p_lower = df['pvalor']-df['u_pvalor']
    alpha_lista = df['alpha']
    a_upper = df['alpha']+df['u_alpha']
    a_lower = df['alpha']-df['u_alpha']
    rango = df['rango']
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Eje para pvalor
    ax1.set_ylim(0, ysupp)
    ax1.plot(rango, pvalor_lista, label="P-value", color=palette[0], linewidth=1)   
    ax1.axhline(y=0.05, color=palette[1], linestyle='--', label='P=0.05', zorder =0, linewidth=1.5)
    ax1.axhline(y= c, color=palette[3], linestyle='--', label= "P=0.20", zorder = 0, linewidth=1.5)
    ax1.fill_between(rango, p_upper, p_lower, alpha=0.4, color=palette[0])
    ax1.set_xlabel("$S_t$", fontsize=35)
    ax1.set_ylabel("P-value", color=palette[0], fontsize=35)
    
    ax1.tick_params(axis='y', labelcolor=palette[0])
    ax1.tick_params(axis='x', which='major', labelsize=30, length=10)
    ax1.tick_params(axis='x', which='minor', labelsize=19, length=5)
    ax1.tick_params(axis='y', which='major', labelsize=30, length=10)
    ax1.tick_params(axis='y', which='minor', labelsize=19, length=5)
    ax1.tick_params(axis='x', pad=10)
    ax1.tick_params(axis='y', pad=10)


    
    # Eje para alpha
    ax2 = ax1.twinx()  
    ax2.set_ylim(0, ysup)
    ax2.plot(rango, alpha_lista, label='$\\alpha_{MLE}$', color=palette[2], linewidth=1.5)
    ax2.fill_between(rango, a_upper, a_lower, alpha=0.4, color=palette[2])
    ax2.set_ylabel("$\\alpha_{MLE}$", color=palette[2], fontsize=35)
    ax2.tick_params(axis='y', labelcolor=palette[2])
    ax2.tick_params(axis='y', which='major', labelsize=30, length=10)
    ax2.tick_params(axis='y', which='minor', labelsize=19, length=5)
    ax2.tick_params(axis='y', pad=10)
    
    # Añadir la leyenda y título
    #plt.title(f"Power law fit with KS test for different lower thresholds - {canal}")
    lines = ax1.get_lines() + ax2.get_lines()  # Combina las líneas de ambos ejes
    labels = [line.get_label() for line in lines]  # Obtiene las etiquetas de todas las líneas
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray', 
               framealpha=0.9, fancybox=True, shadow=False, fontsize=23)
    fig.tight_layout()
    plt.savefig(f"pvalor_alfa_{canal}.pdf", dpi=300)
    plt.show()
    
def opt_alpha (data, k, b=25, ysup=3, ysupp=3, canal='MR', c=0.2, inc=0.01):
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
        if counter % 10 == 0:  # Cada 10 pasos
            print(f'{counter}/{len(rango)}')
            
        for j in range(k):
            data_mont, alpha_s = montecarlo_power_law (i, r, N, alpha_i)
            dsj = ks_montecarlo(data_i, i, r, alpha_s, data_mont)
            ds.append(dsj)
            alpha_m.append(alpha_s)
        # Calculamos la incertidumbre de alfa con montecarlo también
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
# %% Alpha pvalor test MR
dataMR_S = np.array(aval_MR['S'])
maxMR = 30
df_pvalorMR = opt_alpha(dataMR_S, k=500, b = maxMR, c=0.2, inc=0.01)
np.save("df_pvalorMR.npy", df_pvalorMR)
# %% Alpha pvalor test MOKE
dataMO_S = np.array(aval_MOKE['S'])
maxMO = 0.3
df_pvalorMO = opt_alpha(dataMO_S, k = 200, b = 0.3, ysup=4, canal='MOKE', c=0.2, inc=0.0002)
np.save("df_pvalorMO.npy", df_pvalorMO)
# %% Alpha pvalor test MOKE biv
Lbi = np.load("Aval_biv_lista.npy")
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMO = aval_bi['MOKE']
maxMObi = 0.3

df_pvalorMO_bi = opt_alpha(aval_biMO, k=200, b = maxMObi, ysup=3, canal='MOKE', c=0.2, inc=0.0002)
np.save("df_pvalorMO_bi.npy", df_pvalorMO_bi)
# %% Alpla pvalor test MR biv
Lbi = np.load("Aval_biv_lista.npy")
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMR = aval_bi['MR']
maxMRbi = 25

df_pvalorMR_bi = opt_alpha(aval_biMR, k=500, b = maxMRbi, ysup=3, canal='MR', c=0.2, inc=0.05)
np.save("df_pvalorMR_bi.npy", df_pvalorMR_bi)
# %% MR Datos B
dataMRB_S = np.array(aval_MR_B['S'])
maxMRB = 12
df_pvalorMRB = opt_alpha(dataMRB_S, k=300, ysup=12, ysupp=3, b = maxMRB, c=0.2, inc=0.05)
#np.save("df_pvalorMRB.npy", df_pvalorMRB)
# %% MOKE datos B
dataMOB_S = -np.array(aval_MOKE_B['S'])
maxMOB = 0.10
df_pvalorMOB = opt_alpha(dataMOB_S, k=100, b = maxMOB, ysup=5, ysupp=3, c=0.2, canal='MOKE', inc=0.0002)
#np.save("df_pvalorMOB.npy", df_pvalorMOB)

# %% Alpla pvalor test MR biv B
Lbi = np.load("Aval_biv_lista_B.npy")
aval_biB = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMRB = aval_biB['MR']
maxMRBbi = 14

df_pvalorMRB_bi = opt_alpha(aval_biMRB, k=200, b = maxMRBbi, ysup=10, canal='MR', c=0.2, inc=0.05)
#np.save("df_pvalorMRB_bi.npy", df_pvalorMRB_bi)
# %% Alpla pvalor test MO biv B
Lbi = np.load("Aval_biv_lista_B.npy")
aval_biB = pd.DataFrame(Lbi, columns=['MOKE', 'MR'])
aval_biMOB = -aval_biB['MOKE']
maxMOBbi = 0.08

df_pvalorMOB_bi = opt_alpha(aval_biMOB, k=100, b = maxMOBbi, ysup=5, canal='MOKE', c=0.2, inc=0.0002)
#np.save("df_pvalorMOB_bi.npy", df_pvalorMOB_bi)
# %% Cargar archivos elección de alpha

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

# %% Recrear gráficos
pvalor_plot(df_pvalorMR_bi, 3, 'MR')
pvalor_plot(df_pvalorMO_bi, 2.5, 'MOKE')
# %%
pvalor_plot(df_pvalorMR, 5, 'MR')
pvalor_plot(df_pvalorMO, 3, 'MOKE')

# %%
pvalor_plot(df_pvalorMRB, 12, 'MR')
pvalor_plot(df_pvalorMOB, 5, 'MOKE')
# %% Parámetros ajustados para las leyes de potencias
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
# %%
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
# %% Gráficos parámetros ajustados MOKE B
S_power_law(dataMOB_S, minMOB, maxMOB, aMOB, titulo='Canal MOKE', pdf='survival_ajuste_MOB')
plot_power_law(dataMOB_S, minMOB, maxMOB, aMOB, titulo='Canal MOKE', pdf='pdf_ajuste_MOB')
# %% Gráficos parámetros ajustados MR B
S_power_law(dataMRB_S, minMRB, maxMRB, aMRB, titulo='Canal MR', pdf='survival_ajuste_MRB')
plot_power_law(dataMRB_S, minMRB, maxMRB, aMRB, titulo='Canal MR', pdf='pdf_ajuste_MRB')
# %% Gráficos parámetros ajustados MOKE bi
S_power_law(aval_biMO, minMObi,  maxMObi , aMObi , titulo='MOKE', pdf='survival_ajuste_MO_bi')
plot_power_law(aval_biMO, minMObi,  maxMObi, aMObi, titulo='MOKE', pdf='pdf_ajuste_MO_bi')
# %% Gráficos parámetros ajustados MR bi
S_power_law(aval_biMR, minMRbi,  maxMRbi , aMRbi , titulo='MR', pdf='survival_ajuste_MR_bi')
plot_power_law(aval_biMR, minMRbi,  maxMRbi, aMRbi, titulo='MR', pdf='pdf_ajuste_MR_bi')
# %% Gráficos parámetros ajustados MR
dataMR_S = aval_MR['S']
S_power_law(dataMR_S, minMR, maxMR, aMR, titulo='MR', pdf='survival_ajuste_MR')
plot_power_law(dataMR_S, minMR, maxMR, aMR, titulo='MR', pdf='pdf_ajuste_MR')
# %% Gráficos parámetros ajustados MOKE
dataMO_S = aval_MOKE['S']
S_power_law(dataMO_S, minMO, maxMO, aMO, titulo='Canal MOKE', pdf='survival_ajuste_MO')
plot_power_law(dataMO_S, minMO, maxMO, aMO, titulo='Canal MOKE', pdf='pdf_ajuste_MO')
# %% Generar datos bivariadas
MR = pd.read_csv('aval_MR_N.csv')
MO = pd.read_csv('aval_MOKE_N.csv')
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#%%
MO = pd.read_csv('aval_MOKE.csv')
MR = pd.read_csv('aval_MR.csv')
rango = [i for i in range(0, 202) if i != 165] # Archivo 165 es erróneo
#%%
MR = pd.read_csv('aval_MR_B.csv')
MO = pd.read_csv('aval_MOKE_B.csv')
rango=range(0,182)
#%%
MR = pd.read_csv('aval_MR_B_N.csv')
MO = pd.read_csv('aval_MOKE_B_N.csv')
rango=range(0,182)
#%%
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
# %%
aval_bi['MOKE'] = -aval_bi['MOKE']
# %% Gráfico Distribución bivariada
#Lbi = np.load("Aval_biv_lista_B.npy")
#Lbi = np.load('Aval_biv_lista.npy')
aval_bi = pd.DataFrame(Lbi, columns=['MOKE', 'MR']) # Datos originales
#aval_bi['MOKE'] = -aval_bi['MOKE']
aval_bi2 = aval_bi # Para scatter plot
aval_bi = aval_bi[(aval_bi['MOKE']>= minMObi) & (aval_bi['MOKE']<= maxMObi)].reset_index(drop=True) # MR vs MOKE y promedio
aval_bi = aval_bi[(aval_bi['MR']>= minMRbi) & (aval_bi['MR']<= maxMRbi)].reset_index(drop=True)

X = np.log(aval_bi['MOKE']).values
Y = np.log(aval_bi['MR']).values
Xp = np.log(aval_bi['MOKE']).values.reshape(-1, 1)
Xp = sm.add_constant(Xp)

# Definimos nuevas listas para hacer el ajuste MOKE vs MR
Yp = np.log(aval_bi['MR']).values.reshape(-1, 1)
Yp = sm.add_constant(Yp)

model1 = sm.OLS(Y,Xp)
results1 = model1.fit()
Y_pred = results1.predict(Xp)
b1, m1 = results1.params
ub1, um1 = results1.bse
uY = abs(Y_pred) * np.sqrt((X * um1) ** 2 + ub1 ** 2)
upper_bound1 = np.exp(Y_pred) + uY
lower_bound1 = np.exp(Y_pred) - uY

model2 = sm.OLS(X,Yp)
results2 = model2.fit()
X_pred = results2.predict(Yp)
b2, m2 = results2.params
ub2, um2 = results2.bse
uX = abs(X_pred) * np.sqrt((Y * um2) ** 2 + ub2 ** 2)
upper_bound2 = np.exp(X_pred) + uX
lower_bound2 = np.exp(X_pred) - uX

plt.figure(figsize=(14,8))
palette = sns.color_palette("muted")

sns.scatterplot(data=aval_bi2, x='MOKE', y='MR', label='Exp. data', alpha=0.3, color=palette[0])
sns.scatterplot(data=aval_bi, x='MOKE', y='MR', label='Filtered data', alpha=0.3, color=palette[4])
sns.lineplot(x=aval_bi['MOKE'], y=np.exp(Y_pred), label='MR vs MOKE', color=palette[1], linewidth=1.5)
sns.lineplot(y=aval_bi['MR'], x=np.exp(X_pred), label='MOKE vs MR', color=palette[2], linewidth=1.5)

sorted_indices1 = np.argsort(aval_bi['MOKE'])
sorted_indices2 = np.argsort(aval_bi['MR'])

# Aplicar el orden para usar fill_between
x_sorted = aval_bi['MOKE'][sorted_indices1]
y_lower_sorted = lower_bound1[sorted_indices1]
y_upper_sorted = upper_bound1[sorted_indices1]
plt.fill_between(x_sorted, y_upper_sorted, y_lower_sorted, alpha=0.6, color=palette[1])

y_sorted = aval_bi['MR'][sorted_indices2]
x_lower_sorted = lower_bound2[sorted_indices2]
x_upper_sorted = upper_bound2[sorted_indices2]
#plt.fill_betweenx(y_sorted, x_upper_sorted, x_lower_sorted, alpha=0.6, color=palette[2])

# Ajuste promedio
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

#plt.ylim(0.1,)
plt.xlim(0.004,)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Avalanche Size (MR)', fontsize=23, labelpad=10)
plt.xlabel('Avalanche Size (MOKE)', fontsize=23)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='x', which='major', length=10)
plt.tick_params(axis='x', which='minor', length=3)
plt.tick_params(axis='y', which='major', length=10)
plt.tick_params(axis='y', which='minor', length=3) 

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder =0)

plt.legend(loc='lower right', fontsize=18, frameon=True, facecolor='white', 
           edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False)
plt.savefig('bivariada_dist_N.png')

plt.show()

# %%


