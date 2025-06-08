# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:17:23 2025

@author: iorti
"""

# %% Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import statsmodels.api as sm

plt.style.use('science')
palette = sns.color_palette("muted")

MR = pd.read_csv("aval_MR_N.csv")
MO = pd.read_csv("aval_MOKE_N.csv")

# %% Exploration
# %%% Creation of file

rango = [i for i in range(0, 202) if i != 165] # File 165 is incorrect
ajuste_MR = pd.DataFrame([[0,0,0,0]], columns=['M', 'm', 'b', 'Archivo'])

for i in rango:
    MRi = MR[MR['Archivo']==i]
    l = len(MRi)
    url = f'Data_A/hysteresis_deg_{i}.dat' 
    columnas = ['Corriente', 'MOKE', 'MR']
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['t']=range(len(df))
    df = df[df['Corriente']<=-0.025141]
    minM = min(df['MOKE'])
    df['M'] = (df['MOKE'] - minM) / (df['MOKE'].max() - minM)
    for j in range(1,l):
        tii = MRi['ti'].iloc[j]
        tif = MRi['tf'].iloc[j-1]
        dfj = df[(df['t']>= tif)&(df['t']<= tii)]
        Mj = dfj['M'].mean()
        xj = dfj['t'].values.reshape(-1, 1)
        yj = dfj['MR'].values
        # Linear regression fit
        reg = LinearRegression().fit(xj, yj)
        
        # Parameters
        pendiente = reg.coef_[0]
        ordenada = reg.intercept_
                
        new_entry = pd.DataFrame([[Mj, pendiente, ordenada, i]], columns=['M', 'm','b', 'Archivo'])
        ajuste_MR = pd.concat([ajuste_MR, new_entry], ignore_index=True)
ajuste_MR = ajuste_MR.iloc[1:].reset_index(drop=True)

ajuste_MR.to_csv('pendientes_A.csv', index=False)

# %% Prediction model
# %%% Constant fit

pendientes_A = pd.read_csv('pendientes_A.csv')

x = pendientes_A['M'].values
y = pendientes_A['m'].values

y_avg = np.mean(y)
y_std = np.std(y)
x_fit = np.linspace(min(x), max(x), 100)
print(f"Media: {y_avg:.4f}")
print(f"Desviación estándar: {y_std:.4f}")

plt.figure(figsize=(10,8))
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)

plt.scatter(x, y, color=palette[0], alpha=0.3, label='Data', zorder=1)
plt.plot(x_fit, [y_avg]*len(x_fit), '--', color=palette[1], linewidth=2, zorder=3)

plt.fill_between(x_fit, y_avg - y_std, y_avg + y_std, color=palette[1], alpha=0.45,
    label=r'$m = \langle m \rangle \pm \mathrm{std}(m)$', zorder=2)
plt.xlabel(r'$\langle M \rangle$', fontsize=35)
plt.ylabel('m', fontsize=35)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(-0.13,0.65)
plt.xlim(0,)

plt.tick_params(axis='x', which='major', length=12)
plt.tick_params(axis='x', which='minor', length=4)
plt.tick_params(axis='y', which='major', length=12)
plt.tick_params(axis='y', which='minor', length=4)
#plt.title('Slope between avalanches versus Magnetization')
plt.legend(frameon=True,facecolor='white',edgecolor='lightgrey',framealpha=0.9,
    loc='best',fancybox=True,shadow=False, fontsize=28)

plt.savefig('pendientesA.pdf')
plt.show()

# %%% Joint slopes plot

fig, axes = plt.subplots(1, 2, figsize=(20, 8.5), sharey=True)

for i, ax in enumerate(axes):
    if i == 0:
        datos = pd.read_csv('pendientes_A.csv')
    else:
        datos = pd.read_csv('pendientes_B.csv')
    
    x = datos['M'].values
    y = datos['m'].values
    y_avg = np.mean(y)
    y_std = np.std(y)
    x_fit = np.linspace(min(x), max(x), 100)
    ax.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
    ax.scatter(x, y, color=palette[0], alpha=0.3, label='Data', zorder=1)
    ax.plot(x_fit, [y_avg]*len(x_fit), '--', color=palette[1], linewidth=2, zorder=3)
    ax.fill_between(x_fit, y_avg - y_std, y_avg + y_std, color=palette[1], alpha=0.45,
                    label=r'$m = \langle m \rangle \pm \mathrm{std}(m)$', zorder=2)

    ax.set_xlim(min(x_fit)-0.05, max(x_fit)+0.05)
    ax.set_ylim(-0.13, 0.75)
    for label in ax.get_xticklabels():
        label.set_fontsize(30)
    
    for label in ax.get_yticklabels():
        label.set_fontsize(30)

    ax.tick_params(axis='x', which='major', length=12)
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='y', which='major', length=12)
    ax.tick_params(axis='y', which='minor', length=4)

    if i == 0:
        ax.set_ylabel('m', fontsize=35)
        ax.set_xlabel(r'$\langle MOKE \rangle $', fontsize=35, labelpad=10)
        for label in ax.get_yticklabels():
            label.set_fontsize(30)    
        ax.set_title('Dataset A', fontsize=38)
    else:
        ax.set_xlabel(r'$\langle MOKE \rangle$', fontsize=35, labelpad=10)
        ax.set_ylabel('')
        #ax.set_xticks(np.arange(-20, -181, -40)) 
        #ax.set_xticks(np.arange(-20, -180, -8), minor=True)
        ax.set_title('Dataset B', fontsize=38)

    ax.legend(loc='upper right', fontsize=32,
          frameon=True, facecolor='white', edgecolor='lightgrey', framealpha=0.9)
plt.tight_layout()
plt.savefig('pendientes_conjunto.pdf')
plt.show()

# %%% Prediction function

def crear_pred_m ():
    '''

    Returns
    -------
    pred_m : TYPE
        Prediction dataframe with initial anf final time, MR signal and file index.

    '''
    rango = [i for i in range(0, 202) if i != 165] # file 165 is incorrect
    columnas = ['Corriente', 'MOKE', 'MR']
    pred_m = pd.DataFrame([[0,0,0,0]], columns=['ti', 'tf','MR', 'Archivo'])
    
    for j in rango:
        if j % 50 == 0:
            print(j)
        predj = []
        url = f'Data_A/hysteresis_deg_{j}.dat'   
        df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
        df['t']=range(len(df))        
        df = df[df['Corriente']<=-0.025141]
        minM = min(df['MOKE'])
        df['M'] = (df['MOKE'] - minM) / (df['MOKE'].max() - minM)
        
        for i in range(df.index[0]+10,len(df['t'])):
            MRpj = df['MR'][i]
            df_aux = df[(df['t'] >= i - 10) & (df['t'] <= i)]
            
            x = df_aux['t'].values
            y1 = df_aux['MR'].values
            y2 = df_aux['M'].values

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
            
            m_ajuste = 0.0352
            if (m1+std_m1<m_ajuste)&(M<0.8):#&(M>12)&(M<42):
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
                new_entry = pd.DataFrame([[inicio, final, MRp, j]], columns=['ti', 'tf','MR', 'Archivo'])
                pred_m = pd.concat([pred_m, new_entry], ignore_index=True)
                k = 1 # We reset the index to mark a new end
            l -= 1      
    
    pred_m = pred_m.iloc[1:].reset_index(drop=True)
    return pred_m

# %% Plot avalanches and precursors
def aval_plot(avMR, avMO, avP, i):
    '''

    Parameters
    ----------
    avMR : TYPE
        MR avalanches dataframe.
    avMO : TYPE
        MOKE avalanches dataframe.
    avP : TYPE
        Precursors dataframe.
    i : TYPE
        File index.

    Returns
    -------
    None.
    
    Creates a plot with the MR and MOKE signal.
    Flags avalanches and precursors.

    '''
    url = f'Data_A/hysteresis_deg_{i}.dat' 
    columnas = ['Corriente', 'MOKE', 'MR']
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['t']=range(len(df))        
    df = df[df['Corriente']<=-0.025141]
    minM = min(df['MOKE'])
    df['M'] = (df['MOKE'] - minM) / (df['MOKE'].max() - minM)
    avMR = avMR[avMR['Archivo']==i]
    avMO = avMO[avMO['Archivo']==i]
    avP = avP[avP['Archivo']==i]
    
    #df2 = df[(df['MOKE']>12)&(df['MOKE']<42)]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Second y axis

    # All data
    p1, = ax1.plot(df['t'], df['M'], label='MOKE', color=palette[0], marker='o', markersize=2)
    p2, = ax2.plot(df['t'], df['MR'], label='MR', color=palette[3], marker='x', markersize=2)
    #p6, = ax2.plot(df2['t'], df2['MR'], color=palette[4], marker='x', markersize=2, label='Critical region')

    # Avalanches
    p3, = ax2.plot(avMR['ti'], df['MR'][avMR['ti']], label='Avalanches MR', color=palette[2], marker='x', markersize=8, linestyle='None')
    p4, = ax1.plot(avMO['ti'], df['M'][avMO['ti']], label='Avalanches MOKE', color=palette[2], marker='s', markerfacecolor='none', markersize=8, linestyle='None')
    
    # precursors
    p5, = ax2.plot(avP['tf'], avP['MR'], label='Precursors MR', color=palette[4], marker='s', markerfacecolor='none', markersize=8, linestyle='None')

    # Labels and format
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('MOKE', color=palette[0], fontsize=14)
    ax2.set_ylabel('MR', color=palette[3],  fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)

    #ax1.set_xlim(750, 920)
    #ax2.set_ylim(48,52)
    #ax1.set_title('MOKE and MR vs Time')
    #ax1.grid(True)

    # Legend below the figure
    plots = [p1, p2, p3, p4, p5]#, p6]
    labels = [p.get_label() for p in plots]
    ax1.legend(plots,labels,loc='lower right',frameon=True,facecolor='white',
    edgecolor='lightgrey',framealpha=0.9,fancybox=True,shadow=False,  fontsize=14)
    ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
    plt.savefig(f'predictores_{i}.pdf')
    plt.show()


# %% Metrics
# %%% Precursors metrics function

def verificar_prediccion (df, df_aval, ventana=4):
    
    '''

    Parameters
    ----------
    df : TYPE
        Precursors Dataframe.
    df_aval : TYPE
        MR avalanches Dataframe.
    ventana : TYPE, optional
        DESCRIPTION. The default is 4. 
        Length of the window to verify the occurrence of an avalanche after a precursor. 

    Returns
    -------
    list
        [0] = Number of Avalanches detected.
        [1] = Number of False Positives.
        [2] = Precision,
        [3] = Number of actual avalanches.
        
    '''
    l = len(df)
    df['Result']=0
    df_aval['M-Aux'] = 0
    for i in range(l):
        t = df['tf'][i]
        file = df['Archivo'][i]
        df_aux = df_aval[df_aval['Archivo']== file]
        ti = df_aux['ti'].to_list()
        #tf = df_aux['tf'].to_list()

        #for taval, tfaval in zip(ti, tf):  # Recorre cada valor de la lista de avalanchas para ese archivo
        #  if (t + ventana >= taval) and (t <= tfaval):  # Verificas si ti_aval está en el rango [t+1, t+ventana]
        for taval in ti: # Loop through each value in the list of avalanches for that file
            if t  <= taval <= t + ventana: 
                df.at[i, 'Result'] = 1
                df_aval.loc[(df_aval['ti'] == taval) & (df_aval['Archivo'] == file), 'M-Aux'] = 1
    A = df_aval['M-Aux'].sum()
    FP = l - df['Result'].sum()
    TE = 1-FP/l # Success rate (TE='Tasa de éxito')
    return ([A,FP, TE, l])

# %%%% Metrics - Dataset A

pred_m_A = crear_pred_m()
#pred_m_A.to_csv('pred_m_A.csv', index=False)
#pred_m_A = pd.read_csv('pred_m_A.csv')
metricas_m = verificar_prediccion(pred_m_A, MR)
print(metricas_m)

# %% Confusion matrix function
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

def plot_custom_conf_matrix(ax, TP, FP, FN, title=''):
    '''
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    TP : TYPE
        Number of True Positives.
    FP : TYPE
        Number of False Positives.
    FN : TYPE
        Number of False Negatives.
    title : TYPE, optional
        Title of the plot. The default is ''.

    Returns
    -------
    None.

    '''
    # Metrics
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Values and labels
    matrix = np.array([
        [TP, FN],
        [FP, 0]
    ], dtype=float)

    labels = np.array([
        [f'TP = {TP}', f'FN = {FN}'],
        [f'FP = {FP}', 'TN = —']
    ])

    # Colors
    vmax_forzado = np.nanmax(matrix) * 1.5
    norm = mcolors.Normalize(vmin=0, vmax=vmax_forzado)
    cmap = plt.cm.Blues
    # background
    for i in range(2):
        for j in range(2):
            color = cmap(norm(matrix[i, j])) if not (i == 1 and j == 1) else '#eeeeee'
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=color))

    # Text
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, 1 - i + 0.5, labels[i, j],
                    ha='center', va='center', fontsize=35, weight='bold', color='black')

    ax.plot([0, 2], [2, 2], color='black', lw=2)
    ax.plot([0, 2], [0, 0], color='black', lw=2)
    ax.plot([0, 0], [0, 2], color='black', lw=2)  
    ax.plot([2, 2], [0, 2], color='black', lw=2)  

    # inner lines (to divide cells)
    ax.plot([1, 1], [0, 2], color='black', lw=1)
    ax.plot([0, 2], [1, 1], color='black', lw=1)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.2, 1, 'True label', fontsize=35, ha='center', va='center', rotation='vertical')    
    ax.text(1, -0.3, 'Predicted label', ha='center', va='center', fontsize=35)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Center positions of the cells
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    
    # labels
    ax.set_xticklabels(['1', '0'], fontsize=25)
    ax.set_yticklabels(['0', '1'], fontsize=25)
    
    # Display ticks only on the left and bottom axes
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, length=10, width=2, direction='out', color='black')
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, length=10, width=2, direction='out', color='black')
    
    # Desactivate extra ticks
    ax.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
    ax.text(0.6, 2.1,f'{title}', fontsize=40)
    
    # Statistical results
    #ax.text(0.3, -0.35, f'P = {precision:.2f}', ha='center', fontsize=30)
    #ax.text(1.0, -0.35, f'R = {recall:.2f}', ha='center', fontsize=30)
    #ax.text(1.7, -0.35, f'F1 = {f1:.2f}', ha='center', fontsize=30)

# %%% Plots confusion matrix

MR = pd.read_csv("aval_MR_N.csv")
pred_m_A = pd.read_csv('pred_m_A.csv')
metricas_mA = verificar_prediccion(pred_m_A, MR)
FPA = metricas_mA[1]
TPA = metricas_mA[0]
NA = len(MR)
FNA = NA-TPA

MRB = pd.read_csv("aval_MR_B_N.csv")
pred_m_B = pd.read_csv('pred_m_B.csv')
metricas_mB = verificar_prediccion(pred_m_B, MRB)
FPB = metricas_mB[1]
TPB = metricas_mB[0]
NB = len(MRB)
FNB = NB-TPB

# Define cmap and cmap for the bar

cmap = cm.Blues
vmax_forzado = max(np.nanmax([[TPA, FNA], [FPA, np.nan]]),
                   np.nanmax([[TPB, FNB], [FPB, np.nan]])) * 1.5
norm = mcolors.Normalize(vmin=0, vmax=vmax_forzado)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

plot_custom_conf_matrix(axes[0], TP=TPA, FP=FPA, FN=FNA, title='Dataset A')
plot_custom_conf_matrix(axes[1], TP=TPB, FP=FPB, FN=FNB, title='Dataset B')

# Create an additional axis for the colorbar (independent of the subplots)
from matplotlib.cm import ScalarMappable

# Axis coordinates: [left, bottom, width, height]
cbar_ax = fig.add_axes([0.93, 0.135, 0.05, 0.68]) # adjust
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)  #  adjust
cbar.ax.tick_params(labelsize=25, length=12, width=1)
cbar.ax.tick_params(which='minor', length=4, width=1, direction='in')

plt.tight_layout(rect=[0, 0, 0.91, 1])  # leaves space behind
plt.savefig('confussion_matrix.pdf')
plt.show()

# %% Avalanches and precursors
# %%% Read files
MR = pd.read_csv("aval_MR_N.csv")
MO = pd.read_csv("aval_MOKE_N.csv")
MRB = pd.read_csv("aval_MR_B_N.csv")
MOB = pd.read_csv("aval_MOKE_B_N.csv")
pred_m_A = pd.read_csv('pred_m_A.csv')
pred_m_B = pd.read_csv('pred_m_B.csv')
# %%% Plot function
# Manual adjustment of axis limits is required to avoid clutter
# Adjust limits based on the files

def aval_plot_conjunto(avMR, avMO, avP, i, avMRB, avMOB, avPB, j):
    '''
    
    Parameters
    ----------
    avMR : TYPE
        MR Dataset A Avalanches dataframe.
    avMO : TYPE
        MOKE Dataset A Avalanches dataframe.
    avP : TYPE
        Dataset A Precursors dataframe.
    i : TYPE
        Dataset A file index.
    avMRB : TYPE
        MR Dataset B Avalanches dataframe..
    avMOB : TYPE
        MOKE Dataset B Avalanches dataframe..
    avPB : TYPE
        Dataset B Precursors dataframe..
    j : TYPE
        Dataset B file index.

    Returns
    -------
    None.
    
    Plot of the left: Dataset A - File with index i
    Plot of the right: Dataset B - File with index j

    The MR and MOKE signals are displayed, with avalanches and precursors marked in both plots.    
    Joint legend.
    
    '''
    # --- Data left subplot ---
    url_i = f'Data_A/hysteresis_deg_{i}.dat' 
    columnas = ['Corriente', 'MOKE', 'MR']
    df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)
    df_i['t'] = range(len(df_i))   
    df_i = df_i[df_i['Corriente']<=-0.026]
    df_i['MOKE'] = (df_i['MOKE'] - df_i['MOKE'].min()) / (df_i['MOKE'].max() - df_i['MOKE'].min())     
    avMR = avMR[avMR['Archivo'] == i]
    avMO = avMO[avMO['Archivo'] == i]
    avP = avP[avP['Archivo'] == i]

    # --- Data right subplot---
    url_j = f'Data_B/hysteresis_deg_{j}.dat' 
    columnas = ['Corriente', 'MOKE','Hall', 'MR']
    df_j = pd.read_csv(url_j, delim_whitespace=True, header=None, names=columnas)
    df_j['t'] = range(len(df_j)) 
    df_j =df_j[df_j['Corriente']>=0.036290] 
    df_j['MOKE'] = (df_j['MOKE'] - df_j['MOKE'].min()) / (df_j['MOKE'].max() - df_j['MOKE'].min())       
    avMRB = avMRB[avMRB['Archivo'] == j]
    avMOB = avMOB[avMOB['Archivo'] == j]
    avPB = avPB[avPB['Archivo'] == j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), sharey=False)
    fig.subplots_adjust(bottom=0.2, wspace=0.2)

    # --- Left subplot (normal) ---
    ax1_twin = ax1.twinx()
    p1, = ax1.plot(df_i['Corriente'], df_i['MOKE'], label='MOKE', color=palette[0], marker='o', markersize=2)
    p2, = ax1_twin.plot(df_i['Corriente'], df_i['MR'], label='MR', color=palette[3], marker='o', markersize=2)
    p3, = ax1_twin.plot(df_i['Corriente'][avMR['ti']], df_i['MR'][avMR['ti']], label='Avalanches',markeredgewidth=1.5, color=palette[2], marker='s', markersize=12, linestyle='None', markerfacecolor='none')
    p4, = ax1.plot(df_i['Corriente'][avMO['ti']], df_i['MOKE'][avMO['ti']], markeredgewidth=1.5, color=palette[2], marker='s', markerfacecolor='none', markersize=12, linestyle='None')
    p5, = ax1_twin.plot(df_i['Corriente'][avP['tf']], avP['MR'], label='Precursors', color=palette[4],markeredgewidth=1.5, marker='^', markerfacecolor='none', markersize=12, linestyle='None')
    ax1_twin.set_ylim(-24, 50)
    ax1.set_xlim(-0.0322,-0.0259)
    ax1.invert_xaxis()
        
    # --- Right subplot (MR in the center, MOKE to the right) ---
    ax2_twin = ax2.twinx()
    ax2.plot(df_j['Corriente'], df_j['MR'], color=palette[3], marker='o', markersize=2)
    ax2_twin.plot(df_j['Corriente'], df_j['MOKE'], color=palette[0], marker='o', markersize=2)
    ax2.plot(df_j['Corriente'][avMRB['ti']], df_j['MR'][avMRB['ti']], color=palette[2], marker='s',markeredgewidth=1.5, markersize=12, linestyle='None', markerfacecolor='none')
    ax2_twin.plot(df_j['Corriente'][avMOB['ti']], df_j['MOKE'][avMOB['ti']], color=palette[2], markeredgewidth=1.5,marker='s', markerfacecolor='none', markersize=12, linestyle='None')
    ax2.plot(df_j['Corriente'][avPB['tf']], avPB['MR'], color=palette[4], marker='^', markeredgewidth=1.5,markerfacecolor='none', markersize=12, linestyle='None')

    for ax in [ax1, ax2]:
        ax.set_xlabel('$V (V)$', fontsize=35, labelpad=10)
        ax.tick_params(axis='x', labelsize=30)
        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
        if ax == ax1:
            ax.set_title('Dataset A', fontsize=40)
        else:
            ax.set_title('Dataset B', fontsize=40)
    ax1.set_ylabel('MOKE', color=palette[0], fontsize=35, labelpad=15)
    ax1_twin.set_ylabel('MR ($\mu$V)', color=palette[3], fontsize=35, labelpad=-5)
    ax2.set_ylabel('', fontsize=35)
    ax2.set_xlim(0.0405,0.051)
    ax2_twin.invert_yaxis()
    ax2.set_ylim(-74,0)
    ax2_twin.set_ylabel('MOKE', color=palette[0], fontsize=35, labelpad=15)

    ax1.tick_params(axis='y', labelsize=30)
    ax1_twin.tick_params(axis='y', labelsize=30)
    ax2.tick_params(axis='y', labelsize=30)
    ax2_twin.tick_params(axis='y', labelsize=30)
    for ax in [ax1, ax2, ax1_twin, ax2_twin]:
        ax.tick_params(axis='both',which='major', length=12)
        ax.tick_params(axis='both', which='minor', length=5)
        
    plots = [p1, p2, p3, p5]
    labels = [p.get_label() for p in plots]
    fig.legend(plots, labels, loc='lower center', ncol=5, frameon=True, facecolor='white',
               edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False, fontsize=32,
               bbox_to_anchor=(0.5, -0.0))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f'predictores_doble_{i}_{j}.pdf')
    plt.show()
# %%% Plot
aval_plot_conjunto(MR, MO, pred_m_A, 46, MRB, MOB, pred_m_B, 142)

# %% Null method
# %%% Simulations function

from scipy.stats import gaussian_kde
import random 
from collections import Counter

def metodo_nulo(k, media, sd, df_aval, rmin=39, rmax=1040, ventana=4, N=201, exp='A'):
    '''

    Parameters
    ----------
    k : TYPE
        Number of iterations.
    media : TYPE
        Average of precursors per file.
    sd : TYPE
        Standard deviation of precursors per file.
    df_aval : TYPE
        MR Avalanches Dataframe.
    rmin : TYPE, optional
        Lower threshold for prediction. The default is 39.
    rmax : TYPE, optional
        Upper threshold for detction. The default is 1040. (1540 for Dataset B)
    ventana : TYPE, optional
        Window to detect avalanches after each prediction.. The default is 4.
    N : TYPE, optional
        Number of files in the Dataset: 201 for A and 182 for B. The default is 201.
    exp : TYPE, optional
        Dataset: 'A' or 'B'. The default is 'A'.

    Returns
    -------
    A_lista : TYPE
        List with number of avalanches detected in each iteration.
    FP_lista : TYPE
        List with number of false positives detected in each iteration..
    TE_lista : TYPE
        List with precision in each iteration..

    '''
    A_lista = []
    FP_lista = []
    TE_lista = []

    rango_min = rmin
    rango_max = rmax - ventana

    for rep in range(k):  # rep = number of iteration
        # Generates number of precursors per file
        
        num_pred = np.random.normal(loc=media, scale=sd, size=N)
        num_pred = np.round(num_pred).astype(int)
        num_pred = np.clip(num_pred, 0, None)

        # Random times in each file
        resultado = [
            random.sample(range(rango_min, rango_max + 1), n) if n > 0 else []
            for n in num_pred
        ]

        # Adjust number of files
        if exp == 'A':
            archivos = [j if j < 165 else j + 1 for j in range(N)]
        else:
            archivos = list(range(N))

        # Build the Dataframe
        datos = [
            (t, archivos[j]) for j, lista_tiempos in enumerate(resultado) for t in lista_tiempos
        ]
        df_k = pd.DataFrame(datos, columns=["tf", "Archivo"])

        # call the verification function
        A, FP, TE, l = verificar_prediccion(df_k, df_aval)
        A_lista.append(A)
        FP_lista.append(FP)
        TE_lista.append(TE)

        if rep % 50 == 0:
            print(f"Repetición {rep} completada.")

    return A_lista, FP_lista, TE_lista

# %%% Create files

#aval_MR = pd.read_csv('aval_MR_N.csv') # for A
#df_pred = pd.read_csv('pred_m_A.csv') # for A
aval_MR = pd.read_csv('aval_MR_B_N.csv') # for B
df_pred = pd.read_csv('pred_m_B.csv') # for B

df_pred_Archivo = df_pred['Archivo'].to_list()
contador = Counter(df_pred_Archivo)
num_aval = list(contador.values())
media =  np.mean(num_aval)
sd = np.std(num_aval, ddof=0) # ddof=0 es desviación estándar poblacional

#A_nulo, FP_nulo, TE_nulo = metodo_nulo (1000, media, sd, aval_MR) # for A
A_nulo, FP_nulo, TE_nulo = metodo_nulo (1000, media, sd, aval_MR, rmax=1540, N=182, exp='B') # for B
l= len(aval_MR)
A_nulo = A_nulo / l
print(np.mean(np.array(TE_nulo)))
# %%% Save/load files

#np.save("FP_nulo_MR_N.npy", FP_nulo)
#np.save("A_nulo_MR_N.npy", A_nulo)
#np.save("TE_nulo_MR_N.npy", TE_nulo)

#np.save("FP_nulo_MR_N_B.npy", FP_nulo)
#np.save("A_nulo_MR_N_B.npy", A_nulo)
#np.save("TE_nulo_MR_N_B.npy", TE_nulo)

FP_nulo = np.load("FP_nulo_MR_N.npy")
A_nulo = np.load("A_nulo_MR_N.npy")
TE_nulo = np.load("TE_nulo_MR_N.npy")

#FP_nulo = np.load("FP_nulo_MR_N_B.npy")
#A_nulo = np.load("A_nulo_MR_N_B.npy")
#TE_nulo = np.load("TE_nulo_MR_N_B.npy")
# %%% Precision plot

# Histogram with smooth pdf
# Pdf estimation with KDE

kde = gaussian_kde(TE_nulo)
x_vals = np.linspace(min(TE_nulo), max(TE_nulo), 1000)  # Range
y_vals = kde(x_vals)
mean_val = np.mean(TE_nulo)

# Quantiles
q_low, q_high = np.percentile(TE_nulo, [2.5, 97.5])

# Histogram

plt.figure(figsize=(8, 6), dpi=300)
n, bins, patches = plt.hist(TE_nulo, bins=30, density=True, alpha=0.6, color=palette[0], label='$f(x)$')

# Adjust the transparency of each bar
for patch, bin_edge in zip(patches, bins[:-1]):  
    if q_low <= bin_edge <= q_high:
        patch.set_alpha(0.35)  # Darker for 95% confidence interval
    else:
        patch.set_alpha(0.2)

# KDE plot
plt.plot(x_vals, y_vals, color=palette[0], label="KDE", linewidth=2)

# Average line
plt.axvline(x=mean_val, color=palette[1], linestyle='-', linewidth=1.5, label='Mean')
#plt.axvline(x=TE1, color=palette[1], linestyle='--', label=f'SR = {TE1:.2f}')

# Labels and axis
plt.xlabel("Precision", fontsize=30, labelpad=10)
plt.ylabel("$f(x)$", fontsize=30, labelpad=15)
plt.tick_params(axis='both',which='major',length=10,labelsize =25)
plt.tick_params(axis='both',which='minor',length=4,labelsize=25)

plt.legend(loc='upper right',frameon=True,facecolor='white',
edgecolor='lightgrey',framealpha=0.9,fancybox=True,shadow=False,  fontsize=20)
plt.savefig('Nulo_P_Hist_A.pdf')
#plt.savefig('Nulo_P_Hist_B.pdf')
plt.show()

# %%% Recall plot

# Pdf estimation with KDE

kde = gaussian_kde(A_nulo)
x_vals = np.linspace(min(A_nulo), max(A_nulo), 1000)  # Range
y_vals = kde(x_vals)
mean_val = np.mean(A_nulo)

# Quantiles
q_low, q_high = np.percentile(A_nulo, [2.5, 97.5])

# Histogram

plt.figure(figsize=(8, 6), dpi=300)
n, bins, patches = plt.hist(A_nulo, bins=30, density=True, alpha=0.6, color=palette[0], label='$f(x)$')

# Adjust the transparency of each bar
for patch, bin_edge in zip(patches, bins[:-1]):  
    if q_low <= bin_edge <= q_high:
        patch.set_alpha(0.35)  # Darker for 95% confidence interval
    else:
        patch.set_alpha(0.2)

# KDE plot
plt.plot(x_vals, y_vals, color=palette[0], label="KDE", linewidth=2)

# Average line
plt.axvline(x=mean_val, color=palette[1], linestyle='-', linewidth=1.5, label='Mean')
#plt.axvline(x=AV1, color=palette[1], linestyle='--', label=f'Avalanches = {AV1:.0f}')

# Labels and axis
plt.xlabel("Recall", fontsize=30, labelpad=10)
plt.ylabel("$f(x)$", fontsize=30, labelpad=15)
plt.tick_params(axis='both',which='major',length=10,labelsize =25)
plt.tick_params(axis='both',which='minor',length=4,labelsize=25)

#plt.title("Probability density function - Avalanches")
plt.legend(loc='upper right',frameon=True,facecolor='white',
edgecolor='lightgrey',framealpha=0.9,fancybox=True,shadow=False,  fontsize=20)
plt.savefig('Nulo_AV_Hist_A.pdf')
#plt.savefig('Nulo_AV_Hist_B.pdf')
plt.show()