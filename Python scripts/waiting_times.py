# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:40:24 2025

@author: iorti
"""

# %% Librerías
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

# %%
CATP = pd.read_csv('CATP.csv')
#CATP = pd.read_csv('CATP_B.csv')

# %%
CATP0 = CATP[CATP['I']==0]
CATP1 = CATP[CATP['I']==1]
CATP = CATP.sort_values(by=['Archivo', 'ti']).reset_index(drop=True)
CATP0 = CATP0.sort_values(by=['Archivo', 'ti']).reset_index(drop=True)
CATP1 = CATP1.sort_values(by=['Archivo', 'ti']).reset_index(drop=True)

# %% Waiting times thresholds
a = 0.1 # mínimo 
b = 50   # máximo

# Create 50 equally spaced bins on a logarithmic scale between a and b
log_intervals = np.logspace(np.log10(a), np.log10(b), num=50)

# View the generated values
print(log_intervals)

# %% Waiting times for CATP0

waiting_times_dict = {}
url_i = 'Datos/hysteresis_deg_1.dat' 
#url_i='Datos3/hysteresis_deg_1.dat'
columnas = ['Corriente', 'MOKE','MR']
#columnas=['Corriente','MOKE','Hall','MR']
df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)

# Iterate over the thresholds in the log_intervals
for log_threshold in log_intervals:
    waiting_times = []
    CATP0_filtered = CATP0[CATP0['S_MR'] >= log_threshold] # Filtrar

    # Order by 'Archivo' y 'ti'
    CATP0_sorted = CATP0_filtered.sort_values(by=['Archivo', 'ti'])

    # Compute the waiting times between consecutive rows within the same 'File'
    for i in range(len(CATP0_sorted) - 1):
        # Make sure we are within the same file
        if CATP0_sorted['Archivo'].iloc[i] == CATP0_sorted['Archivo'].iloc[i + 1]:
            # Compute the waiting time between 'tf' of the current row and 'ti' of the next row
            #waiting_time = CATP0_sorted['ti'].iloc[i + 1] - CATP0_sorted['tf'].iloc[i]
            waiting_time = np.abs(df_i['Corriente'][CATP0_sorted['ti'].iloc[i+1]]-df_i['Corriente'][CATP0_sorted['tf'].iloc[i]])
            waiting_times.append(waiting_time)

    waiting_times_dict[log_threshold] = waiting_times

max_length = max(len(times) for times in waiting_times_dict.values())

# Fill the shorter lists with NaN
for key in waiting_times_dict:
    waiting_times_dict[key].extend([np.nan] * (max_length - len(waiting_times_dict[key])))
# Create a DataFrame with the results
waiting_times_df0 = pd.DataFrame(waiting_times_dict)
waiting_times_df0.columns = [f'DIFF{i+1}' for i in range(waiting_times_df0.shape[1])]
# Convert to integers
waiting_times_df0 = waiting_times_df0.fillna(0)
#waiting_times_df0 = waiting_times_df0.astype(int)
# Print the dataframe
print(waiting_times_df0)

# %% Waiting times for CATP1

waiting_times_dict = {}

# Iterate over the thresholds in the log_intervals
for log_threshold in log_intervals:
    waiting_times = []
    CATP1_filtered = CATP1[CATP1['S_MR'] >= log_threshold] # Filtrar

    # Order by 'Archivo' y 'ti'
    CATP1_sorted = CATP1_filtered.sort_values(by=['Archivo', 'ti'])

    for i in range(len(CATP1_sorted) - 1):
        # Make sure we are within the same file
        if CATP1_sorted['Archivo'].iloc[i] == CATP1_sorted['Archivo'].iloc[i + 1]:
            # Compute the waiting time between 'tf' of the current row and 'ti' of the next row
            waiting_time = CATP1_sorted['ti'].iloc[i + 1] - CATP1_sorted['tf'].iloc[i]
            waiting_times.append(waiting_time)
            
    waiting_times_dict[log_threshold] = waiting_times

max_length = max(len(times) for times in waiting_times_dict.values())

# Fill the shorter lists with NaN
for key in waiting_times_dict:
    waiting_times_dict[key].extend([np.nan] * (max_length - len(waiting_times_dict[key])))
# Create a DataFrame with the results
waiting_times_df1 = pd.DataFrame(waiting_times_dict)
waiting_times_df1.columns = [f'DIFF{i+1}' for i in range(waiting_times_df1.shape[1])]
# Convert to integers
waiting_times_df1 = waiting_times_df1.fillna(0)
waiting_times_df1 = waiting_times_df1.astype(int)
# Print the dataframe
print(waiting_times_df1)

# %% Waiting times for CATP

waiting_times_dict = {}

# Iterate over the thresholds in the log_intervals
for log_threshold in log_intervals:
    waiting_times = []
    CATP_filtered = CATP[CATP['S_MR'] >= log_threshold] # Filtrar

    # Order by 'Archivo' y 'ti'
    CATP_sorted = CATP_filtered.sort_values(by=['Archivo', 'ti'])

    for i in range(len(CATP_sorted) - 1):
        # Make sure we are within the same file
        if CATP_sorted['Archivo'].iloc[i] == CATP_sorted['Archivo'].iloc[i + 1]:
            # Compute the waiting time between 'tf' of the current row and 'ti' of the next row
            waiting_time = CATP_sorted['ti'].iloc[i + 1] - CATP_sorted['tf'].iloc[i]
            waiting_times.append(waiting_time)

    waiting_times_dict[log_threshold] = waiting_times

max_length = max(len(times) for times in waiting_times_dict.values())

# Fill the shorter lists with NaN
for key in waiting_times_dict:
    waiting_times_dict[key].extend([np.nan] * (max_length - len(waiting_times_dict[key])))
# Create a DataFrame with the results
waiting_times_df = pd.DataFrame(waiting_times_dict)
waiting_times_df.columns = [f'DIFF{i+1}' for i in range(waiting_times_df.shape[1])]
# Convert to integers
waiting_times_df = waiting_times_df.fillna(0)
waiting_times_df = waiting_times_df.astype(int)
# Print the dataframe
print(waiting_times_df)

# %% Save files
#waiting_times_df.to_csv("WT.csv", index = False)
#waiting_times_df0.to_csv("WT0V.csv", index = False)
#waiting_times_df1.to_csv("WT1.csv", index = False)

#waiting_times_df.to_csv("WT_B.csv", index = False)
#waiting_times_df0.to_csv("WT0V_B.csv", index = False)
#waiting_times_df1.to_csv("WT1_B.csv", index = False)
# %% Load files
WT = pd.read_csv('WT.csv')
WT0 = pd.read_csv('WT0V.csv')
WT1 = pd.read_csv('WT1.csv')

#WT = pd.read_csv('WT_B.csv')
#WT0 = pd.read_csv('WT0V_B.csv')
#WT1 = pd.read_csv('WT1_B.csv')

# %% Pdf waiting times
from scipy.special import factorial

def plot_dist_WT(df, i, num_bins=10, num_bootstrap=10000, q=0.16):
    threshold = log_intervals[i-1]
    col_name = f'DIFF{i}'
    data = df[col_name]
    data = data[data > 0]
       
    min_val = data.min()
    max_val = data.max()
    
    # Generate bins using logspace
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
    # Calculate the normalised histogram (empirical density function)
    hist, bin_edges = np.histogram(data, bins=bin_edges, density=True)

    # Compute bins centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Bootstrap
    densb = np.zeros((num_bootstrap, len(bin_centers)))
    for b in range(num_bootstrap):
        xi_b = np.random.choice(data, size=len(data), replace=True)  # Sampling with replacement
        xi_b[xi_b < min_val] = min_val  # Adjust values out of range
        xi_b[xi_b > max_val] = max_val
        z_b = np.histogram(xi_b, bins=bin_edges, density=True)  # bootstrap histogram
        densb[b, :] = z_b[0]  # Save the density

    # Confidence intervals
    ci_lower = np.percentile(densb, 100 * q, axis=0)
    ci_upper = np.percentile(densb, 100 * (1 - q), axis=0)

    plt.figure(figsize=(8, 6))

    plt.errorbar(bin_centers, hist, yerr=[hist - ci_lower, ci_upper - hist], 
                 label=f'Threshold {threshold:.2f} WT Density', fmt='o', color=palette[0], linestyle='-', 
                 markersize=5, capsize=3)

    # Labels and title
    plt.xlabel('Waiting Time')
    plt.ylabel('Probability density function')
    plt.title(f'Empirical Density Function of Threshold - {threshold:.2f}')
    plt.xscale('log') # log-log scale
    plt.yscale('log')
    # Legend
    plt.legend()
    plt.tight_layout()
    # Show plot
    #plt.savefig(f'dist_WT_B_{threshold:.2f}.pdf')
    plt.show()

# %% 
plot_dist_WT(WT0, 15)
# %% First version of reescalated pdf
# El objetivo de este gráfico es comparar la distribución de WT con un Poisson
def plot_dist_escalado_WT(df, i, k, num_bins=10, num_bootstrap=10000, q=0.16):
    threshold = log_intervals[i-1]
    col_name = f'DIFF{i}'
    data = df[col_name]
    data = data[data > 0]
    lambda_e = np.mean(data)

    min_val = data.min()
    max_val = data.max()
    data = data.sort_values()
    y = np.exp(-data/lambda_e)

    # Generar los bins usando logspace
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
    # Calcular el histograma normalizado (función de densidad empírica)
    hist, bin_edges = np.histogram(data, bins=bin_edges, density=True)

    # Calcular los centros de los bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = bin_centers/lambda_e
    hist = hist * lambda_e
    # Bootstrap
    densb = np.zeros((num_bootstrap, len(bin_centers)))
    for b in range(num_bootstrap):
        xi_b = np.random.choice(data, size=len(data), replace=True)  # Muestreo con reemplazo
        xi_b[xi_b < min_val] = min_val  # Ajuste de valores fuera del rango
        xi_b[xi_b > max_val] = max_val
        z_b = np.histogram(xi_b, bins=bin_edges, density=True)  # Histograma bootstrap
        densb[b, :] = z_b[0]  # Guardar la densidad

    # Intervalos de confianza
    ci_lower = np.percentile(densb, 100 * q, axis=0)
    ci_upper = np.percentile(densb, 100 * (1 - q), axis=0)
    plt.errorbar(bin_centers, hist, yerr=[hist - lambda_e*ci_lower, lambda_e*ci_upper - hist], 
                 label=f'$S_t$ = {threshold:.2f}', fmt='o', color=palette[(k+8) % len(palette)], linestyle='-', 
                 markersize=5, capsize=3, linewidth=2)
    plt.plot(data/lambda_e, y, linestyle='--', color=palette[(k+8) % len(palette)])
    
# %% Segunda versión: cajas representan los bins en vez de plot puntual
# El objetivo de este gráfico es comparar la distribución de WT con un Poisson

def plot_dist_escalado_WT(df, i, k, num_bins=10, num_bootstrap=10000, q=0.16):
    threshold = log_intervals[i-1]
    col_name = f'DIFF{i}'
    data = df[col_name]
    data = data[data > 0]
    lambda_e = np.mean(data)

    min_val = data.min()
    max_val = data.max()
    data = data.sort_values()
    y = np.exp(-data / lambda_e)

    # Bins logarítmicos
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
    hist, _ = np.histogram(data, bins=bin_edges, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = bin_centers / lambda_e
    hist = hist * lambda_e
    bin_edges_scaled = bin_edges / lambda_e

    # Bootstrap
    densb = np.zeros((num_bootstrap, len(bin_centers)))
    for b in range(num_bootstrap):
        xi_b = np.random.choice(data, size=len(data), replace=True)
        xi_b = np.clip(xi_b, min_val, max_val)
        z_b = np.histogram(xi_b, bins=bin_edges, density=True)
        densb[b, :] = z_b[0]

    # Intervalos de confianza escalados
    ci_lower = np.percentile(densb, 100 * q, axis=0) * lambda_e
    ci_upper = np.percentile(densb, 100 * (1 - q), axis=0) * lambda_e

    color = palette[(k + 8) % len(palette)]

    # Dibujar los rectángulos como bins con incertidumbre
    for left, right, yc, ylow, yhigh in zip(bin_edges_scaled[:-1], bin_edges_scaled[1:], hist, ci_lower, ci_upper):
        if yc > 0:
            plt.fill_between([left, right], [ylow, ylow], [yhigh, yhigh], color=color, alpha=0.4, edgecolor=None)
            plt.hlines(yc, left, right, color=color, lw=2)
    plt.plot(bin_centers, hist, color=color, linewidth=2)
    # Curva exponencial
    plt.plot(data / lambda_e, y, linestyle='-', color=color, label=f'$S_t$ = {threshold:.2f}')

# %% Plot de una selección de Pdf reescaladas de waiting time thresholds

plt.figure(figsize=(12,8))
k = 1
for i in range(1,30,3):
    k += 1
    plot_dist_escalado_WT(WT0, i, k)

plt.xlabel(r'$\frac{t}{\langle \Delta V \rangle}$',fontsize=30)
plt.ylabel(r'$\mathbb{P}(\Delta V) \langle \Delta V \rangle$',fontsize=30, labelpad=10)
#plt.title (f'Empirical Density Function for different thresholds')
plt.xscale('log') # Escala log-log
plt.yscale('log')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder =0)
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5), fontsize=25,frameon=True, facecolor='white', 
           edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False)
plt.tight_layout()
#plt.savefig('dist_WT_comparacion_B2.pdf')
plt.show()

# %% Gráfico comparativa distribuciones possion conjunto
# Plot de una selección de Pdf reescaladas de waiting time thresholds

WT0 = pd.read_csv('WT0V.csv')
WT0_B = pd.read_csv('WT0V_B.csv')

fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True, sharey=True)

archivos = [
    {'WT': WT0_B, 'title': 'Dataset B (filtrado)'},
    {'WT': WT0,   'title': 'Dataset original'}
]

# Para recoger leyendas comunes
all_handles = []
all_labels = []

for ax, archivo in zip(axes, archivos):
    k = 1
    plt.sca(ax)
    for i in range(1, 30, 3):
        k += 1
        plot_dist_escalado_WT(archivo['WT'], i, k)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if ax == axes[0]:
        ax.set_ylabel(r'$\mathbb{P}(\Delta V) \langle \Delta V \rangle$', fontsize=35, labelpad=20)
        ax.set_title('Dataset A', fontsize=40)
    else:
        ax.set_title('Dataset B', fontsize=40)      
    ax.set_xlabel(r'$\frac{\Delta V}{\langle \Delta V \rangle}$', fontsize=50, labelpad=10)
    ax.tick_params(axis='both', labelsize=35, pad=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_ylim(bottom=1e-5)
    ax.set_xlim(left=2e-2)
    ax.tick_params(axis='both', which='major', length=10)
    ax.tick_params(axis='both', which='minor', length=3)
    
    x_exp = np.logspace(-2, 2, 500)
    y_exp = np.exp(-x_exp)
    ax.plot(x_exp, y_exp, 'k--', linewidth=2)
    # Capturar leyenda solo del primer eje
    handles, labels = ax.get_legend_handles_labels()
    if not all_labels:
        all_handles = handles
        all_labels = labels
        
# Leyenda común debajo
fig.legend(all_handles, all_labels, loc='lower center', ncol=5, fontsize=30,
           frameon=True, facecolor='white', edgecolor='lightgrey', framealpha=0.9, bbox_to_anchor=(0.534,-0.16))

plt.tight_layout(w_pad=3)
plt.savefig('dist_WT_conjunto.pdf')
plt.show()

# %% Count  waiting times
plt.figure(figsize=(12,8))
k = 0
for i in range(1,41, 5):
    threshold = log_intervals[i-1]
    col_name = f'DIFF{i}'
    WTi = WT0[col_name]
    N = len(WTi[WTi>0])
    WTi = WTi[WTi>0].value_counts().reset_index()
    WTi.columns = ['valor', 'frecuencia']
    WTi = WTi.sort_values(by='valor')
    WTi['acum'] = WTi['frecuencia'].cumsum()
    plt.step(WTi['valor'], WTi['acum'], where='post', color=palette[k % len(palette)],marker='o',
    markersize=1.5, linewidth=1.5, label =f'T = {threshold:.2f}, N = {N} ')
    k += 1
plt.xlabel('$\Delta V (V)$', fontsize=35, labelpad=10)
plt.ylabel('Count WT', fontsize=35, labelpad=12)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(0,470)
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=24,
           frameon=True, facecolor='white', edgecolor='lightgrey',
           framealpha=0.9, fancybox=True, shadow=False)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
#plt.savefig('escalated_WT_sl.pdf')
plt.show()

# %% CDF WT gráfico individual
plt.figure(figsize=(10,8))
k = 0
for i in range(1, 41, 5):
    threshold = log_intervals[i-1]
    col_name = f'DIFF{i}'
    WTi = WT0[col_name]
    WTi = WTi[WTi > 0]
    N = len(WTi)
    
    WTi = WTi.value_counts().reset_index()
    WTi.columns = ['valor', 'frecuencia']
    WTi = WTi.sort_values(by='valor')
    WTi['acum'] = WTi['frecuencia'].cumsum()
    
    # Normalizar para obtener CDF
    WTi['cdf'] = WTi['acum'] / N

    plt.step(WTi['valor'], WTi['cdf'], where='post',
             color=palette[k % len(palette)],
             marker='o', markersize=1.5, linewidth=1.5,
             label=f'T = {threshold:.2f}, N = {N}')
    k += 1

plt.xlabel('Size of waiting time', fontsize=35, labelpad=10)
plt.ylabel('Cummulative distribution', fontsize=35, labelpad=12)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
plt.tight_layout()
plt.legend(fontsize=25, frameon=True, facecolor='white',
           edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False)
#plt.savefig('CDF_WT_sl.pdf')
plt.show()
# %% Gráfico conjunto CDF waiting times
# Cargar los datos
WT0 = pd.read_csv('WT0V.csv')
WT0_B = pd.read_csv('WT0V_B.csv')

# Crear la figura y los subgráficos
fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=True)  # Dos subgráficos lado a lado

# Para el gráfico de la izquierda
url_i = 'Datos/hysteresis_deg_1.dat' 
columnas = ['Corriente', 'MOKE', 'MR']
df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)
k = 0

for i in range(1, 40, 4):
    threshold = log_intervals[i-1]
    col_name = f'DIFF{i}'
    WTi = WT0[col_name]
    WTi = WTi[WTi > 0]
    N = len(WTi)
    
    WTi = WTi.value_counts().reset_index()
    WTi.columns = ['valor', 'frecuencia']
    WTi = WTi.sort_values(by='valor')
    WTi['acum'] = WTi['frecuencia'].cumsum()
    
    # Normalizar para obtener CDF
    WTi['cdf'] = WTi['acum'] / N

    axes[0].step(WTi['valor'], WTi['cdf'], where='post',
                 color=palette[k],  # Usamos el color de la paleta
                 marker='o', markersize=1.5, linewidth=1.5,
                 label=f'$S_t$ = {threshold:.2f}')
    k += 1

axes[0].set_xlabel('Inter-event $|\Delta V| (V)$', fontsize=35, labelpad=12)
axes[0].set_ylabel('Cumulative probability', fontsize=35, labelpad=14)
axes[0].tick_params(axis='both', labelsize=30, which='major', length=12)
axes[0].tick_params(axis='both', which='minor', length=4)
axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
axes[0].set_title('Dataset A', fontsize=40)
# Para el gráfico de la derecha
k = 0
for i in range(1, 40, 4):
    threshold = log_intervals[i-1]
    col_name = f'DIFF{i}'
    WTi = WT0_B[col_name]
    WTi = WTi[WTi > 0]
    N = len(WTi)
    
    WTi = WTi.value_counts().reset_index()
    WTi.columns = ['valor', 'frecuencia']
    WTi = WTi.sort_values(by='valor')
    WTi['acum'] = WTi['frecuencia'].cumsum()
    
    # Normalizar para obtener CDF
    WTi['cdf'] = WTi['acum'] / N

    axes[1].step(WTi['valor'], WTi['cdf'], where='post',
                 color=palette[k],  # Usamos el mismo color de la paleta
                 marker='o', markersize=1.5, linewidth=1.5)
    k += 1

axes[1].set_xlabel('Inter-event $|\Delta V| (V)$', fontsize=35, labelpad=12)
#axes[1].set_ylabel('CDF of waiting times', fontsize=20, labelpad=12)
axes[1].tick_params(axis='both', labelsize=30, which='major', length=12)
axes[1].tick_params(axis='both', which='minor', length=4)
axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
axes[1].set_title('Dataset B', fontsize=40)
axes[1].minorticks_on()

# Ajuste del espacio y leyenda
fig.subplots_adjust(bottom=0.2)
fig.legend(loc='upper center', bbox_to_anchor=(0.53, 0.02), ncol=5, fontsize=28, frameon=True, facecolor='white',
           edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False)

# Mostrar y guardar el gráfico
plt.tight_layout()
plt.savefig('CDF_WT_conjunto.pdf')
plt.show()

# %% Evolución N
N = []
for i in range(1,41):
    col_name = f'DIFF{i}'
    WTi = WT0[col_name]
    Ni = len(WTi[WTi>0])
    N.append(Ni)
x = range(len(N))
plt.figure(figsize=(8,6))
plt.plot(log_intervals[0:40],N, marker='o', color=palette[0])
plt.xlabel('Threshold for waiting times')
plt.ylabel('Number of waiting times')
#plt.savefig('evolucion_N_WT_B.pdf')
plt.show()

# %% Test de Bi
CATP = pd.read_csv("CATP_B.csv")
#CATP = pd.read_csv("CATP.csv")
CATP0 = CATP[CATP['I']==0]
#rango = [i for i in range(0, 202) if i != 165]
rango = range(0,182)

plt.figure(figsize=(12,8))
plt.axhline(0, color='black', linewidth = 0.5)  # Línea horizontal en y=0
plt.axhline(1.36, color='gray', linewidth=2, label='$\\alpha = 0.05$')
plt.axhline(-1.36, color='gray', linewidth=2)
plt.axhline(-1.48, color='darkgray', linewidth=2, label='$\\alpha = 0.01$')
plt.axhline(1.48, color='darkgray', linewidth=2)

l = 1
for k in range(0, 30, 3):
    l += 1
    threshold = log_intervals[k]
    inter_event_times = []
    CATP0 = CATP0[CATP0['S_MR']>threshold]
    CATP0_sorted = CATP0.sort_values(by=['Archivo', 'ti'])
    for i in rango:
        CATP0i = CATP0_sorted[CATP0_sorted['Archivo']==i]
        li = len(CATP0i)
        for j in range(1,li-1):
            # Calcular el waiting time entre 'tf' de la fila actual y 'ti' de la siguiente fila
            f_waiting_time = CATP0i['ti'].iloc[j + 1] - CATP0i['tf'].iloc[j]
            waiting_time = CATP0i['ti'].iloc[j] - CATP0i['tf'].iloc[j-1]
            inter_event_times.append([waiting_time,f_waiting_time,i])
            
    bi_times = pd.DataFrame(data=inter_event_times, columns=['WT', 'FWT', 'File'])
    
    bi_times['delta_ti'] = np.minimum(bi_times['WT'],bi_times['FWT'])
    
    bi_times['tau_ti'] = 0
    
    for i in rango:
        bi_i = bi_times[bi_times['File']==i]
        li = len(bi_i)
        for j in range(1,li-1):
            if bi_i['delta_ti'].iloc[j] == bi_i['WT'].iloc[j]:
                bi_times.loc[(bi_times['File'] == i) & (bi_times.index == bi_i.index[j]), 'tau_ti'] = bi_i['WT'].iloc[j-1]
            else:
                bi_times.loc[(bi_times['File'] == i) & (bi_times.index == bi_i.index[j]), 'tau_ti'] = bi_i['FWT'].iloc[j+1]
            
    bi_times = bi_times[bi_times['tau_ti']>0]
    bi_times['H_i'] = bi_times['delta_ti']/(bi_times['delta_ti']+bi_times['tau_ti']/2)
    Hi_ordered = np.array(bi_times['H_i'].sort_values())
    n = len(Hi_ordered)
    FnH = np.arange(1, len(Hi_ordered) + 1) / len(Hi_ordered)
    Deltaf = np.sqrt(n)*(FnH-Hi_ordered)
    Deltaf = np.concatenate(([0], Deltaf, [0]))
    Hi_ordered = np.concatenate(([0], Hi_ordered, [1]))
    plt.plot(Hi_ordered, Deltaf, linewidth=1.5,color=palette[(l+8) % len(palette)], label=f'S = {threshold:.2f}')
    # maxDf = max(Deltaf)
    #print(maxDf)
    
plt.ylabel('$\\Delta f(H)$', fontsize=35)
plt.xlabel('$H_n$', fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.ylim(-1.9,1.9)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30,frameon=True, facecolor='white', 
           edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
plt.tight_layout()
plt.savefig('bi_test_comparacion_B2.pdf')
plt.show()

# %% Test de Bi suavizado
CATP = pd.read_csv("CATP_B.csv")
#CATP = pd.read_csv("CATP.csv")
CATP0 = CATP[CATP['I']==0]
#rango = [i for i in range(0, 202) if i != 165]
rango = range(0,182)

plt.figure(figsize=(12,8))
plt.axhline(0, color='black', linewidth = 0.5)  # Línea horizontal en y=0
plt.axhline(1.36, color='gray', linewidth=2, label='$\\alpha=0.05$')
plt.axhline(-1.36, color='gray', linewidth=2)
plt.axhline(-1.48, color='darkgray', linewidth=2, label='$\\alpha=0.01$')
plt.axhline(1.48, color='darkgray', linewidth=2)

l = 1
for k in range(0, 30, 3):
    l += 1
    threshold = log_intervals[k]
    inter_event_times = []
    CATP0 = CATP0[CATP0['S_MR']>threshold]
    CATP0['ti'] = CATP0['ti'] + np.random.uniform(0, 1, size=len(CATP0))
    CATP0['tf'] = CATP0['tf'] + np.random.uniform(0, 1, size=len(CATP0))
    CATP0_sorted = CATP0.sort_values(by=['Archivo', 'ti'])
    for i in rango:
        CATP0i = CATP0_sorted[CATP0_sorted['Archivo']==i]
        li = len(CATP0i)
        for j in range(1,li-1):
            # Calcular el waiting time entre 'tf' de la fila actual y 'ti' de la siguiente fila
            f_waiting_time = CATP0i['ti'].iloc[j + 1] - CATP0i['tf'].iloc[j]
            waiting_time = CATP0i['ti'].iloc[j] - CATP0i['tf'].iloc[j-1]
            inter_event_times.append([waiting_time,f_waiting_time,i])
            
    bi_times = pd.DataFrame(data=inter_event_times, columns=['WT', 'FWT', 'File'])
    
    bi_times['delta_ti'] = np.minimum(bi_times['WT'],bi_times['FWT'])
    
    bi_times['tau_ti'] = 0
    bi_times['tau_ti'] = bi_times['tau_ti'].astype(float)
    
    for i in rango:
        bi_i = bi_times[bi_times['File']==i]
        li = len(bi_i)
        for j in range(1,li-1):
            if bi_i['delta_ti'].iloc[j] == bi_i['WT'].iloc[j]:
                bi_times.loc[(bi_times['File'] == i) & (bi_times.index == bi_i.index[j]), 'tau_ti'] = bi_i['WT'].iloc[j-1]
            else:
                bi_times.loc[(bi_times['File'] == i) & (bi_times.index == bi_i.index[j]), 'tau_ti'] = bi_i['FWT'].iloc[j+1]
            
    bi_times = bi_times[bi_times['tau_ti']>0]
    bi_times['H_i'] = bi_times['delta_ti']/(bi_times['delta_ti']+bi_times['tau_ti']/2)
    Hi_ordered = np.array(bi_times['H_i'].sort_values())
    n = len(Hi_ordered)
    FnH = np.arange(1, len(Hi_ordered) + 1) / len(Hi_ordered)
    Deltaf = np.sqrt(n)*(FnH-Hi_ordered)
    Deltaf = np.concatenate(([0], Deltaf, [0]))
    Hi_ordered = np.concatenate(([0], Hi_ordered, [1]))
    plt.plot(Hi_ordered, Deltaf, linewidth=1.5,color=palette[(l+8) % len(palette)], label=f'S = {threshold:.2f}')
    # maxDf = max(Deltaf)
    #print(maxDf)
    
plt.ylabel('$\\Delta f(H)$', fontsize=35)
plt.xlabel('$H_n$', fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.ylim(-1.9,1.9)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30,frameon=True, facecolor='white', 
           edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
plt.tight_layout()
plt.savefig('bi_test_suave_comparacion_B2.pdf')
plt.show()

# %% Gráfico Bi Test conjunto
fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=False)

# Definir datasets
datasets = [
    {
        'filename': 'CATP.csv',
        'rango': range(0, 182),
        'title': 'Original dataset (files 0–181)'
    },
    {
        'filename': 'CATP_B.csv',
        'rango': [i for i in range(0, 202) if i != 165],
        'title': 'Dataset B (excl. file 165)'
    }
]

for ax, ds in zip(axes, datasets):
    CATP = pd.read_csv(ds['filename'])
    CATP0 = CATP[CATP['I'] == 0]
    rango = ds['rango']

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(1.36, color='gray', linewidth=3, label=r'$\alpha=0.05$')
    ax.axhline(-1.36, color='gray', linewidth=3)
    ax.axhline(1.48, color='darkgray', linewidth=3, label=r'$\alpha=0.01$')
    ax.axhline(-1.48, color='darkgray', linewidth=3)
    if ax == axes[0]:
        ax.set_title('Dataset A', fontsize=40)
    else:
        ax.set_title('Dataset B', fontsize=40)
    l = 1
    for k in range(0, 30, 3):
        l += 1
        threshold = log_intervals[k]
        CATP0_filtrado = CATP0[CATP0['S_MR'] > threshold].copy()
        CATP0_filtrado = CATP0_filtrado.sort_values(by=['Archivo', 'ti'])
        
        # Escoger ruta y columnas dependiendo del dataset
        if ds['filename'] == 'CATP.csv':
            url_i = f'Datos/hysteresis_deg_1.dat'
            columnas = ['Corriente', 'MOKE', 'MR']
        else:
            url_i = f'Datos3/hysteresis_deg_1.dat'
            columnas = ['Corriente', 'MOKE', 'Hall', 'MR']

        try:
            df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)
        except FileNotFoundError:
            continue  # Saltarse si no existe el archivo

        inter_event_times = []
        for i in rango:
            grupo = CATP0_filtrado[CATP0_filtrado['Archivo'] == i]
            li = len(grupo)
            for j in range(1, li - 1):
                f_waiting_time = np.abs(df_i['Corriente'][grupo['ti'].iloc[j+1]] - df_i['Corriente'][grupo['tf'].iloc[j]])
                waiting_time = np.abs(df_i['Corriente'][grupo['ti'].iloc[j]] - df_i['Corriente'][grupo['tf'].iloc[j-1]])
                inter_event_times.append([waiting_time, f_waiting_time, i])

        bi_times = pd.DataFrame(inter_event_times, columns=['WT', 'FWT', 'File'])
        bi_times['delta_ti'] = np.minimum(bi_times['WT'], bi_times['FWT'])
        bi_times['tau_ti'] = 0.0

        for i in rango:
            bi_i = bi_times[bi_times['File'] == i]
            li = len(bi_i)
            for j in range(1, li - 1):
                idx = bi_i.index[j]
                if bi_i['delta_ti'].iloc[j] == bi_i['WT'].iloc[j]:
                    bi_times.loc[idx, 'tau_ti'] = bi_i['WT'].iloc[j - 1]
                else:
                    bi_times.loc[idx, 'tau_ti'] = bi_i['FWT'].iloc[j + 1]

        bi_times = bi_times[bi_times['tau_ti'] > 0]
        bi_times['H_i'] = bi_times['delta_ti'] / (bi_times['delta_ti'] + bi_times['tau_ti'] / 2)
        Hi_ordered = np.sort(bi_times['H_i'].values)
        n = len(Hi_ordered)
        FnH = np.arange(1, n + 1) / n
        Deltaf = np.sqrt(n) * (FnH - Hi_ordered)
        Hi_ordered = np.concatenate(([0], Hi_ordered, [1]))
        Deltaf = np.concatenate(([0], Deltaf, [0]))

        ax.plot(Hi_ordered, Deltaf, linewidth=1.5, color=palette[(l + 8) % len(palette)], label=f'$S_t$ = {threshold:.2f}')

    ax.set_xlabel(r'$H_n$', fontsize=35)
    ax.set_ylabel(r'$\Delta f(H)$', fontsize=35)
    ax.tick_params(axis='both', labelsize=35)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

# Leyenda común
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=35,columnspacing=0.5,
           frameon=True, facecolor='white', edgecolor='lightgrey', framealpha=0.9, bbox_to_anchor=(0.525, -0.2))

plt.tight_layout(w_pad=2)
plt.savefig('bi_test_conjunto.pdf')
plt.show()

# %% Gráfico Bi Test suavizado conjunto
fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=False)

datasets = [
    {
        'filename': 'CATP.csv',
        'rango': range(0, 182),
        'title': 'Original dataset (files 0–181)'
    },
    {
        'filename': 'CATP_B.csv',
        'rango': [i for i in range(0, 202) if i != 165],
        'title': 'Dataset B (excl. file 165)'
    }
]

for ax, ds in zip(axes, datasets):
    CATP = pd.read_csv(ds['filename'])
    CATP0 = CATP[CATP['I'] == 0]
    rango = ds['rango']

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(1.36, color='gray', linewidth=3, label=r'$\alpha=0.05$')
    ax.axhline(-1.36, color='gray', linewidth=3)
    ax.axhline(1.48, color='darkgray', linewidth=3, label=r'$\alpha=0.01$')
    ax.axhline(-1.48, color='darkgray', linewidth=3)
    if ax == axes[0]:
        ax.set_title('Dataset A', fontsize=40)
    else:
        ax.set_title('Dataset B', fontsize=40)
    
    # Escoger ruta y columnas dependiendo del dataset
    if ds['filename'] == 'CATP.csv':
        url_i = f'Datos/hysteresis_deg_30.dat'
        columnas = ['Corriente', 'MOKE', 'MR']
    else:
        url_i = f'Datos3/hysteresis_deg_30.dat'
        columnas = ['Corriente', 'MOKE', 'Hall', 'MR']

    try:
        df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)
    except FileNotFoundError:
        continue  # Saltarse si no existe el archivo

    l = 1
    for k in range(0, 30, 3):
        l += 1
        threshold = log_intervals[k]
        filtered = CATP0[CATP0['S_MR'] > threshold].copy()
        filtered['Ci'] = df_i['Corriente'].iloc[filtered['ti'].values].values
        filtered['Cf'] = df_i['Corriente'].iloc[filtered['tf'].values].values
        filtered['Ci'] += np.random.uniform(0, 0.000001, size=len(filtered))
        filtered['Cf'] += np.random.uniform(0, 0.000001, size=len(filtered))
        filtered['ti'] += np.random.uniform(0, 1, size=len(filtered))
        filtered['tf'] += np.random.uniform(0, 1, size=len(filtered))
        filtered = filtered.sort_values(by=['Archivo', 'ti'])

        inter_event_times = []
        for i in rango:
            group = filtered[filtered['Archivo'] == i]
            li = len(group)
            for j in range(1, li - 1):
                f_waiting_time = np.abs(group['Ci'].iloc[j + 1] - group['Cf'].iloc[j])
                waiting_time = np.abs(group['Ci'].iloc[j] - group['Cf'].iloc[j - 1])
                inter_event_times.append([waiting_time, f_waiting_time, i])

        bi_times = pd.DataFrame(inter_event_times, columns=['WT', 'FWT', 'File'])
        bi_times['delta_ti'] = np.minimum(bi_times['WT'], bi_times['FWT'])
        bi_times['tau_ti'] = 0.0

        for i in rango:
            bi_i = bi_times[bi_times['File'] == i]
            li = len(bi_i)
            for j in range(1, li - 1):
                idx = bi_i.index[j]
                if bi_i['delta_ti'].iloc[j] == bi_i['WT'].iloc[j]:
                    bi_times.loc[idx, 'tau_ti'] = bi_i['WT'].iloc[j - 1]
                else:
                    bi_times.loc[idx, 'tau_ti'] = bi_i['FWT'].iloc[j + 1]

        bi_times = bi_times[bi_times['tau_ti'] > 0]
        bi_times['H_i'] = bi_times['delta_ti'] / (bi_times['delta_ti'] + bi_times['tau_ti'] / 2)
        Hi_ordered = np.sort(bi_times['H_i'].values)
        n = len(Hi_ordered)
        FnH = np.arange(1, n + 1) / n
        Deltaf = np.sqrt(n) * (FnH - Hi_ordered)
        Hi_ordered = np.concatenate(([0], Hi_ordered, [1]))
        Deltaf = np.concatenate(([0], Deltaf, [0]))

        ax.plot(Hi_ordered, Deltaf, linewidth=1.5, color=palette[(l + 8) % len(palette)], label=f'$S_t$ = {threshold:.2f}')

    ax.set_xlabel(r'$H_n$', fontsize=35)
    ax.set_ylabel(r'$\Delta f(H)$', fontsize=35)
    ax.tick_params(axis='both', labelsize=35)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

# Leyenda común
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=35,columnspacing=0.5,
           frameon=True, facecolor='white', edgecolor='lightgrey', framealpha=0.9, bbox_to_anchor=(0.525, -0.2))

plt.tight_layout(w_pad=2)  # deja espacio abajo para la leyenda
plt.savefig('bi_test_suave_conjunto.pdf')
plt.show()