# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:07:18 2025

@author: iorti
"""

# %% Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
palette = sns.color_palette("muted")

# %% Import catalogue
CATP = pd.read_csv("CATP_B_N.csv")
#CATP = pd.read_csv("CATP.csv")


# %% Counting Process

CATP0 = CATP[CATP['I']==0]
CATP1 = CATP[CATP['I']==1]

CATP0_grouped = CATP0.groupby('ti', as_index=False).agg({  
    'S_MR': 'sum',  
    'S_MO': 'sum',  
    'Ip': 'sum',  
    'T': 'sum',
    'P': 'sum',
})  

# Agregar una nueva columna con el número de datos agrupados  
CATP0_grouped['count'] = CATP0.groupby('ti').size().values

CATP1_grouped = CATP1.groupby('ti', as_index=False).agg({  
    'S_MR': 'sum',  
    'S_MO': 'sum',  
    'Ip': 'sum',   
    'T': 'sum',
})  

CATP1_grouped['count'] = CATP1.groupby('ti').size().values

CATP_grouped = CATP.groupby('ti', as_index=False).agg({  
    'S_MR': 'sum',  
    'S_MO': 'sum',  
    'Ip': 'sum', 
    'T': 'sum',
})  

CATP_grouped['count'] = CATP.groupby('ti').size().values

# %% Counting MR and MO (CATP0)
cols = ['S_MR', 'count', 'S_MO', 'Ip','P', 'T']
labels = ['Size of MR', 'Count of Events', 'Size of MO', 'Precursor intensity','precursor', 'T (MR)']

# Create figure
plt.figure(figsize=(12, 8))

# Graficamos la CDF para cada columna en función de 'ti'
for i, (col, label) in enumerate(zip(cols, labels)):
    sorted_df = CATP0_grouped.sort_values(by='ti')  # Ordenamos por 'ti'
    cumulative = np.cumsum(sorted_df[col]) / np.sum(sorted_df[col])  # Normalizamos
    plt.plot(sorted_df['ti'], cumulative, label=label,linewidth=1.5, color=palette[i])

# Labels and title
plt.xlabel('Time', fontsize=35)
plt.ylabel('Cumulative Distribution', fontsize=35, labelpad=15)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.title('Cumulative Distributions as a function of ti - Avalanches both in MOKE and MR')
plt.tight_layout()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, facecolor='white', edgecolor='lightgrey', framealpha=0.9, fancybox=True, shadow=False, fontsize=30)
plt.subplots_adjust(bottom=0.25)

plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
#plt.savefig('counting_process_B2.pdf')
# Show plot
plt.show()

# %% Joint plot of CATP0 and CATP0_B
cols = ['S_MR', 'count', 'S_MO', 'Ip','P', 'T']
labels = ['S(MR)', 'Count A', 'S(MOKE)', 'Intensity P','Count P', 'Period(MR)']

fig, axs = plt.subplots(1, 2, figsize=(22,9), sharey=True)
CATP = pd.read_csv("CATP_N.csv")
CATP0 = CATP[CATP['I']==0]
CATP0_grouped = CATP0.groupby('ti', as_index=False).agg({  'S_MR': 'sum',  
    'S_MO': 'sum',  'Ip': 'sum', 'P':'sum', 'T': 'sum',})  
CATP0_grouped['count'] = CATP0.groupby('ti').size().values
url_i ='Datos/hysteresis_deg_1.dat' 
columnas = ['Corriente', 'MOKE', 'MR']
df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)
for i, (col, label) in enumerate(zip(cols, labels)):
    sorted_df = CATP0_grouped.sort_values(by='ti')
    cumulative = np.cumsum(sorted_df[col]) / np.sum(sorted_df[col])
    axs[0].plot(df_i['Corriente'][sorted_df['ti']], cumulative, label=label, linewidth=1.5, color=palette[i])
CATP = pd.read_csv("CATP_B_N.csv")
CATP0 = CATP[CATP['I']==0]
CATP0_grouped = CATP0.groupby('ti', as_index=False).agg({  'S_MR': 'sum',  
    'S_MO': 'sum',  'Ip': 'sum','P':'sum', 'T': 'sum',})  
CATP0_grouped['count'] = CATP0.groupby('ti').size().values
url_i ='Datos3/hysteresis_deg_1.dat' 
columnas = ['Corriente', 'MOKE','Hall', 'MR']
df_i = pd.read_csv(url_i, delim_whitespace=True, header=None, names=columnas)
for i, (col, label) in enumerate(zip(cols, labels)):
    sorted_df = CATP0_grouped.sort_values(by='ti')
    cumulative = np.cumsum(sorted_df[col]) / np.sum(sorted_df[col])   
    axs[1].plot(df_i['Corriente'][sorted_df['ti']], cumulative, label=label, linewidth=1.5, color=palette[i])  # Ejemplo: mismo gráfico con otro estilo

# Labels
axs[0].invert_xaxis()
axs[0].set_xlabel('$V (V)$', fontsize=35, labelpad=10)
axs[0].set_ylabel('Cumulative Distribution', fontsize=35, labelpad=15)
axs[1].set_xlabel('$V (V)$', fontsize=35, labelpad=10)
axs[1].tick_params(axis='both', labelsize=30, which='major', length=12)
axs[1].tick_params(axis='both', which='minor', length=4)
axs[0].tick_params(axis='both', labelsize=30, which='major', length=12)
axs[0].tick_params(axis='both', which='minor', length=4)

axs[0].set_xticks(np.arange(-0.032, -0.0259, 0.002))
axs[1].set_xticks(np.arange(0.040, 0.051, 0.005))

for ax in axs:
    ax.tick_params(labelsize=30)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    if ax==axs[0]:
        ax.set_title('Dataset A', fontsize=40)
    else:
        ax.set_title('Dataset B', fontsize=40)

# Shared legend for both plots below
fig.legend(labels, loc='upper center', bbox_to_anchor=(0.535, 0.02), ncol=6, fontsize=32, frameon=True,
           facecolor='white', edgecolor='lightgrey', framealpha=0.9, fancybox=True, columnspacing=0.5)
fig.subplots_adjust(bottom=0.25, wspace=0.05)
fig.tight_layout()

plt.savefig('counting_conjunto.pdf')
plt.show()

# %% Only MO (CATP1)
cols = ['S_MR', 'count', 'S_MO', 'Ip', 'T']
labels = ['Size of MR', 'Count of Events', 'Size of MO', 'Precursor intensity', 'T (MO)']

# Create figure
plt.figure(figsize=(8, 6))

# We plot the CDF for each column as a function of 'ti'
for i, (col, label) in enumerate(zip(cols, labels)):
    sorted_df = CATP1_grouped.sort_values(by='ti')  # Order by 'ti'
    cumulative = np.cumsum(sorted_df[col]) / np.sum(sorted_df[col])  # Normalize
    plt.plot(sorted_df['ti'], cumulative, label=label, color=palette[i])

# Labels and title
plt.xlabel('ti')
plt.ylabel('Cumulative Distribution $\\mathbb{F}(t_i)$')
plt.legend()
plt.title('Cumulative Distributions as a function of ti - Only avalanches in MOKE')
plt.tight_layout()
#plt.savefig('counting_process_MO.pdf')
# Show plot
plt.show()

# %% Complete catalogue (CATP0)
cols = ['S_MR', 'count', 'S_MO', 'Ip', 'T']
labels = ['Size of MR', 'Count of Events', 'Size of MO', 'Precursor intensity', 'T']

# Create figure
plt.figure(figsize=(8, 6))

# We plot the CDF for each column as a function of 'ti'
for i, (col, label) in enumerate(zip(cols, labels)):
    sorted_df = CATP_grouped.sort_values(by='ti')  # Order by 'ti'
    cumulative = np.cumsum(sorted_df[col]) / np.sum(sorted_df[col])  # Normalize
    plt.plot(sorted_df['ti'], cumulative, label=label, color=palette[i])

# Labels and title
plt.xlabel('ti')
plt.ylabel('Cumulative Distribution $\\mathbb{F}(t_i)$')
plt.legend()
plt.title('Cumulative Distributions as a function of ti - Complete Dataset')
plt.tight_layout()
#plt.savefig('counting_process_CATP.pdf')
plt.show()

# %% Rescaling with averages

def plot_reescalado (data):
    t_media = np.mean(data)
    values, counts = np.unique(data, return_counts=True)
    x = values/t_media
    pt = counts/len(data)
    y = t_media*pt
    y_exp = np.exp(-x)
    
    plt.figure(figsize=(8,6))
    plt.plot(x, y, color=palette[0], label=r'$\mathbb{P}(t) \langle t \rangle$')
    plt.plot(x, y_exp, color=palette[1], label=r'$\exp\left(-\frac{t}{\langle t \rangle}\right)$')
    plt.xlabel(r'$\frac{t}{\langle t \rangle}$', fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #plt.title(f'Reescalated to compare with an Homogeneous Poisson - Complete Dataset')
    plt.ylabel('Value', fontsize=28)
    plt.legend(fontsize=28)
    plt.tight_layout()
    #plt.savefig(f'reescalate_poisson_all_B.pdf')
    plt.show()

# %% Rescaling CATP0 (Poisson comparison)
waiting_times = []
CATP0_sorted = CATP0.sort_values(by=['Archivo', 'ti'])
for i in range(len(CATP0_sorted) - 1):
    # Make sure we are in the same file
    if CATP0_sorted['Archivo'].iloc[i] == CATP0_sorted['Archivo'].iloc[i + 1]:
        # Compute the waiting time between the current row's tf and the next row's ti
        waiting_time = CATP0_sorted['ti'].iloc[i + 1] - CATP0_sorted['tf'].iloc[i]
        waiting_times.append(waiting_time)
plot_reescalado(waiting_times)

# %% Waiting times thresholds
a = 0.1 # minimum 
b = 50   # maximum

# Create 50 evenly spaced bins on a logarithmic scale between a and b
log_intervals = np.logspace(np.log10(a), np.log10(b), num=50)

# %%
CATP0_grouped['Acum'] = CATP0_grouped['count'].cumsum()
# %% Waiting times list
#rango = [i for i in range(0, 202) if i != 165]
rango = range(0,182)
L = []
for i in rango:
    L.append(len(CATP0[CATP0['Archivo']==i]))

def insertar_lista(waiting_times, L):
    nueva_lista = []
    pos = 0  # Position in waiting_times
    
    for salto in L:
        nueva_lista.extend(waiting_times[pos:pos + salto])  # Keeps the actual values
        nueva_lista.append(1)  # Inserts 1
        pos += salto # Move the position forward
        
    nueva_lista.extend(waiting_times[pos:])  # Agregates the rest of waiting times
    return nueva_lista
waiting_times1 = insertar_lista(waiting_times, L)
S_acum = np.cumsum(CATP0['S_MR'])
# %% Aftershocks with counting process
Mag = np.digitize(CATP0['S_MR'], log_intervals) - 1
Mag[Mag<20] = 0

waiting_times_acum = np.cumsum(waiting_times1)
plt.figure(figsize=(8,6))
plt.plot(waiting_times_acum, S_acum, color=palette[0], label='Avalanches')

# Convert lists to arrays to facilitate filtering
waiting_times_acum = np.array(waiting_times_acum)
S_acum = np.array(S_acum)
S_MR = np.array(CATP0['S_MR'])  # Convert CATP0['S_MR'] to array

# Filter the data points where S_MR > 4.5
mask = Mag != 0

# Draw circles in the selected points
plt.scatter(waiting_times_acum[mask], S_acum[mask], 
            s=(Mag[mask] / 5)**2.5 , edgecolors='r', facecolors='none', label='Magnitude $> 4$')

plt.xlim(0,15000)
plt.ylim(0,800)
plt.xlabel('Time')
plt.ylabel('Count of Size of MR')
plt.legend(loc='upper left')
#plt.savefig('aftershocks.pdf')
plt.show()

# %% Bi Test

inter_event_times = []
CATP0_sorted = CATP0.sort_values(by=['Archivo', 'ti'])
for i in rango:
    CATP0i = CATP0_sorted[CATP0_sorted['Archivo']==i]
    li = len(CATP0i)
    for j in range(1,li-1):
        # Compute the waiting time between the current row's tf and the next row's ti
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
maxDf = max(Deltaf)
print(maxDf)

# %% Plot: Delta f(H) - Bi Test
plt.figure(figsize=(8,6))
plt.axhline(0, color='black', linewidth = 0.5)  # Horizontal line at y=0
plt.axhline(1.36, color=palette[1], label='$95\%$ confidence')
plt.axhline(-1.36, color=palette[1])
plt.plot(Hi_ordered, Deltaf, color=palette[0], label='$\\Delta f(H)$')
#plt.gca().spines['bottom'].set_position(('data', 0))  # Mueve el eje x a y = 0
#plt.gca().spines['left'].set_position(('data', 0)) 
plt.ylabel('$\\Delta f(H)$')
plt.xlabel('$H_n$')
plt.legend(loc='lower right',bbox_to_anchor=(1, 0.05))
plt.tight_layout()
plt.savefig('bi_test_B.pdf')
plt.show()

