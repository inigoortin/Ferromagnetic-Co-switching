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

# %% Slope and magnetization between avalanches

rango = [i for i in range(0, 202) if i != 165] # File 165 is incorrect
ajuste_MR = pd.DataFrame([[0,0,0,0]], columns=['M', 'm', 'b', 'Archivo'])

for i in rango:
    MRi = MR[MR['Archivo']==i]
    l = len(MRi)
    url = f'Datos/hysteresis_deg_{i}.dat' 
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

# %% Ajuste sigmoide creciente
# Función de crecimiento sigmoide (solo la parte ascendente)
def sigmoide_creciente(x, L, k, x_0):
    return L * (1 - np.exp(-k * (x - x_0)))

# Datos
x = ajuste_MR['M'].values
y = ajuste_MR['m'].values

# Ajuste de la sigmoide creciente
params, _ = curve_fit(sigmoide_creciente, x, y, p0=[max(y), 1, np.median(x)])

# Visualización
x_vals = np.linspace(x.min(), x.max(), 500)
y_fit = sigmoide_creciente(x_vals, *params)

plt.figure(figsize=(8,6))
plt.scatter(x, y, color=palette[0], alpha=0.5, label='Datos')
plt.plot(x_vals, y_fit, '--', color=palette[1], label='Ajuste sigmoide creciente')
plt.xlabel('M')
plt.ylabel('m')
plt.title('Ajuste sigmoide creciente')
plt.legend()
plt.grid(True)
plt.show()

print(f"Parámetros ajustados: L={params[0]:.4f}, k={params[1]:.4f}, x_0={params[2]:.4f}")

# %% Ajuste sigmoide creciente (incertidumbre)
# Función de crecimiento sigmoide (solo la parte ascendente)
def sigmoide_creciente(x, L, k, x_0):
    return L * (1 - np.exp(-k * (x - x_0)))

# Datos
x = ajuste_MR['M'].values
y = ajuste_MR['m'].values

# Ajuste de la sigmoide creciente
params, covariance_matrix = curve_fit(sigmoide_creciente, x, y, p0=[max(y), 1, np.median(x)])
#params, covariance_matrix = curve_fit(sigmoide_gen, x, y, p0=[2,12])

# Calcular incertidumbre (desviación estándar de los parámetros)
uncertainties = np.sqrt(np.diag(covariance_matrix))

# Crear un rango de valores x para dibujar la curva ajustada y sus incertidumbres
x_vals = np.linspace(x.min(), x.max(), 500)
y_fit = sigmoide_creciente(x_vals, *params)

# Calcular las bandas de incertidumbre
y_upper = sigmoide_creciente(x_vals, *(params + uncertainties))
y_lower = sigmoide_creciente(x_vals, *(params - uncertainties))

plt.figure(figsize=(8,6))
# Visualización
plt.scatter(x, y, color=palette[0], alpha=0.3, label='Datos')
plt.plot(x_vals, y_fit, '--', color=palette[1], label='Ajuste sigmoide creciente')
plt.axvline(15, color=palette[3], linewidth=2)

# Banda de incertidumbre
plt.fill_between(x_vals, y_lower, y_upper, color=palette[1], alpha=0.3, label='Incertidumbre del ajuste')

plt.xlabel('M')
plt.ylabel('m')
plt.title('Ajuste sigmoide creciente con incertidumbre')
plt.legend()
#plt.grid(True)
plt.show()

# Mostrar los parámetros ajustados y sus incertidumbres
print("Parámetros ajustados:")
print(f"L = {params[0]:.4f} ± {uncertainties[0]:.4f}")
print(f"k = {params[1]:.4f} ± {uncertainties[1]:.4f}")
print(f"x_0 = {params[2]:.4f} ± {uncertainties[2]:.4f}")

# %% Ajuste sigmoide generalizada
def sigmoide_gen (x,a,b):
    return np.log(a**(x-b)/(1+a**(x-b)))

# Datos
d = ajuste_MR[ajuste_MR['M']>15]
x = d['M'].values
y = d['m'].values

# Ajuste de la sigmoide generalizada
params, _ = curve_fit(sigmoide_gen, x, y, p0=[2,12])

# Visualización
x_vals = np.linspace(x.min(), x.max(), 500)
y_fit = sigmoide_gen(x_vals, *params)

coeffs = np.polyfit(x, y, deg=2)
p = np.poly1d(coeffs)
print("Coeficientes del polinomio ajustado:", coeffs)

# Evaluar el polinomio en un rango suave de x para graficar
x1_fit = np.linspace(15, max(x), 100)
y1_fit = p(x1_fit)

plt.figure(figsize=(8,6))
plt.scatter(x, y, color=palette[0], alpha=0.3, label='Datos')
plt.plot(x1_fit, y1_fit, '--', color=palette[2], label='Ajuste polinómico')

plt.plot(x_vals, y_fit, '--', color=palette[1], label='Ajuste sigmoide generalizada')
plt.axvline(15, color=palette[3], linewidth=2)

# params, _ = curve_fit(sigmoide_creciente, x, y, p0=[max(y), 1, np.median(x)])
# Visualización
# y_fit = sigmoide_creciente(x_vals, *params)
# plt.plot(x_vals, y_fit, '--', color=palette[2], label='Ajuste sigmoide creciente')

plt.xlabel('M')
plt.ylabel('m')
#plt.title('Ajuste sigmoide generalizada')
plt.legend()
plt.show()

print(f"Parámetros ajustados: a={params[0]:.4f}, b={params[1]:.4f}")

# %% Constant fit
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

# %% Joint slopes plot
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
        ax.set_xlabel(r'$\langle M \rangle $', fontsize=35)
        for label in ax.get_yticklabels():
            label.set_fontsize(30)    
        ax.set_title('Dataset A', fontsize=38)
    else:
        ax.set_xlabel(r'$\langle M \rangle$', fontsize=35)
        ax.set_ylabel('')
        #ax.set_xticks(np.arange(-20, -181, -40)) 
        #ax.set_xticks(np.arange(-20, -180, -8), minor=True)
        ax.set_title('Dataset B', fontsize=38)

    ax.legend(loc='upper right', fontsize=32,
          frameon=True, facecolor='white', edgecolor='lightgrey', framealpha=0.9)
plt.tight_layout()
plt.savefig('pendientes_conjunto.pdf')
plt.show()

# %% Prediction with slope method
def crear_pred_m ():
    rango = [i for i in range(0, 202) if i != 165] # file 165 is incorrect
    columnas = ['Corriente', 'MOKE', 'MR']
    pred_m = pd.DataFrame([[0,0,0,0]], columns=['ti', 'tf','MR', 'Archivo'])
    for j in rango:
        if j % 50 == 0:
            print(j)
        predj = []
        url = f'Datos/hysteresis_deg_{j}.dat'   
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
    url = f'Datos/hysteresis_deg_{i}.dat' 
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

# %% Precursors metrics
def verificar_prediccion (df, df_aval, ventana=4):
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

# %% Precursor metrics - data set A
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
    # Cálculos
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Valores y etiquetas
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

    # Posiciones centrales de las celdas
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

# %% Plots confusion matrix
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

# Define cmap y norm para la barra
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

# %%
aval_plot(MR, MO, pred_m_A, 48)
# %% Files to be used in joint plot
MR = pd.read_csv("aval_MR_N.csv")
MO = pd.read_csv("aval_MOKE_N.csv")
MRB = pd.read_csv("aval_MR_B_N.csv")
MOB = pd.read_csv("aval_MOKE_B_N.csv")
pred_m_A = pd.read_csv('pred_m_A.csv')
pred_m_B = pd.read_csv('pred_m_B.csv')
# %% Joint plot of predictors
# Manual adjustment of axis limits is required to avoid clutter
# Adjust limits based on the files

def aval_plot_conjunto(avMR, avMO, avP, i, avMRB, avMOB, avPB, j):
    # --- Data left subplot ---
    url_i = f'Datos/hysteresis_deg_{i}.dat' 
    columnas = ['Corriente', 'MOKE', 'MR']
    df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)
    df_i['t'] = range(len(df_i))   
    df_i['MOKE'] = (df_i['MOKE'] - df_i['MOKE'].min()) / (df_i['MOKE'].max() - df_i['MOKE'].min())     
    avMR = avMR[avMR['Archivo'] == i]
    avMO = avMO[avMO['Archivo'] == i]
    avP = avP[avP['Archivo'] == i]

    # --- Data right subplot---
    url_j = f'Datos3/hysteresis_deg_{j}.dat' 
    columnas = ['Corriente', 'MOKE','Hall', 'MR']
    df_j = pd.read_csv(url_j, delim_whitespace=True, header=None, names=columnas)
    df_j['MOKE'] = (df_j['MOKE'] - df_j['MOKE'].min()) / (df_j['MOKE'].max() - df_j['MOKE'].min())
    df_j['t'] = range(len(df_j))        
    avMRB = avMRB[avMRB['Archivo'] == j]
    avMOB = avMOB[avMOB['Archivo'] == j]
    avPB = avPB[avPB['Archivo'] == j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), sharey=False)
    fig.subplots_adjust(bottom=0.2, wspace=0.2)

    # --- Left subplot (normal) ---
    ax1_twin = ax1.twinx()
    p1, = ax1.plot(df_i['Corriente'], df_i['MOKE'], label='MOKE', color=palette[0], marker='o', markersize=2)
    p2, = ax1_twin.plot(df_i['Corriente'], df_i['MR'], label='MR', color=palette[3], marker='o', markersize=2)
    p3, = ax1_twin.plot(df_i['Corriente'][avMR['ti']], df_i['MR'][avMR['ti']], label='Avalanches',markeredgewidth=2, color=palette[2], marker='s', markersize=14, linestyle='None', markerfacecolor='none')
    p4, = ax1.plot(df_i['Corriente'][avMO['ti']], df_i['MOKE'][avMO['ti']], markeredgewidth=2, color=palette[2], marker='s', markerfacecolor='none', markersize=14, linestyle='None')
    p5, = ax1_twin.plot(df_i['Corriente'][avP['tf']], avP['MR'], label='Precursors', color=palette[4],markeredgewidth=2, marker='^', markerfacecolor='none', markersize=14, linestyle='None')
    #ax1.set_ylim(-4, 47)
    ax1.set_xlim(-0.0322,-0.0259)
    ax1.invert_xaxis()
        
    # --- Right subplot (MR in the center, MOKE to the right) ---
    ax2_twin = ax2.twinx()
    ax2.plot(df_j['Corriente'], df_j['MR'], color=palette[3], marker='o', markersize=2)
    ax2_twin.plot(df_j['Corriente'], df_j['MOKE'], color=palette[0], marker='o', markersize=2)
    ax2.plot(df_j['Corriente'][avMRB['ti']], df_j['MR'][avMRB['ti']], color=palette[2], marker='s',markeredgewidth=2, markersize=14, linestyle='None', markerfacecolor='none')
    ax2_twin.plot(df_j['Corriente'][avMOB['ti']], df_j['MOKE'][avMOB['ti']], color=palette[2], markeredgewidth=2,marker='s', markerfacecolor='none', markersize=14, linestyle='None')
    ax2.plot(df_j['Corriente'][avPB['tf']], avPB['MR'], color=palette[4], marker='^', markeredgewidth=2,markerfacecolor='none', markersize=14, linestyle='None')

    for ax in [ax1, ax2]:
        ax.set_xlabel('$V (V)$', fontsize=35, labelpad=10)
        ax.tick_params(axis='x', labelsize=30)
        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
        if ax == ax1:
            ax.set_title('Dataset A', fontsize=40)
        else:
            ax.set_title('Dataset B', fontsize=40)
    ax1.set_ylabel('MOKE', color=palette[0], fontsize=35, labelpad=10)
    ax1_twin.set_ylabel('MR', color=palette[3], fontsize=35, labelpad=-10)
    ax2.set_ylabel('', fontsize=35)
    ax2.set_xlim(0.0405,0.051)
    ax2_twin.invert_yaxis()
    #ax2_twin.set_ylim(-30, 185)
    ax2_twin.set_ylabel('MOKE', color=palette[0], fontsize=35, labelpad=10)

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
# %%
aval_plot_conjunto(MR, MO, pred_m_A, 44, MRB, MOB, pred_m_B, 142)
