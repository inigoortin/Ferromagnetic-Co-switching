# %% Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import powerlaw
import scienceplots # unified style and LaTex format
plt.style.use('science')
palette = sns.color_palette("muted")

# %% Avalanche detection
# %%% Load file

columnas = ['Corriente', 'MOKE', 'MR']
url = 'Data_A/hysteresis_deg_1.dat' # 190 for zoom precursor plots
df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)

df['dMOKE'] = df['MOKE'].diff(-1) / df['Corriente'].diff(-1)
df['dMR'] = df['MR'].diff(-1) / df['Corriente'].diff(-1)
l = len(df)
df['dMR'][l-1]=0
df['dMOKE'][l-1]=0
df['t']=range(len(df))

# %%% MR Avalanches

df['MR_Filtro']=abs(df['dMR'])>30000
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
# Prints número de avalanchas según el filtro
aval1 = df_MRf.shape[0]
aval2 = df_MRf2.shape[0]
aval3 = df_MRa.shape[0]
print("Las avalanchas tras primer filtrado para el canal MR son", aval1)
print("Las avalanchas que duran más de una unidad de tiempo para el canal MR son", aval2)
print("El número de avalanchas independientes para el canal MR es", aval3)
t_MRa = df_MRa['ti'] # t minúscula porque representa los tiempos de inicio de la avalancha
t_MRf2 = df_MRf2['t']
df_MRa
# %%% MOKE Avalanches

# Dynamic left aligned threshold for the derivatives

ventana = 25 # da mejor resultado que 20 (discrimina mejor)
df['MO_std_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).std() if len(x) > 1 else np.nan, raw=False)
df['MO_media_local'] = df['dMOKE'].rolling(window=ventana, center=False).apply(lambda x: x.drop(x.abs().idxmax()).mean() if len(x) > 1 else np.nan, raw=False)
df['MOKE_umbral'] =2.5*df['MO_std_local']- df['MO_media_local'] #IMP
df['MOKE_Filtro']=df['dMOKE']<-df['MOKE_umbral']
# Se crea un umbral variable de 2.5 distribuciones estándar en una ventana centrada de 50 datos
# Se elimina el máximo en el intervalo para evitar que una avalancha muy grande impida que se reconozcan otras avalanchas
# menores presentes en la muestra.

#print(df.head())
df_MOf = df[df['MOKE_Filtro'] == True] # f de filtro
mask = (df_MOf['t'].diff(-1) == -1) | (df_MOf['t'].diff(1) == 1)
df_MOf2 = df_MOf[mask]
mask2 = df_MOf2['t'].diff(1) != 1
df_MOa = df_MOf2[mask2] # a de avalanchas

df_MOf = df_MOf.copy() # evita warning
df_MOf['S'] = df_MOf.apply(lambda row: df.loc[df['t'] == row['t']+1, 'MOKE'].values[0] - df.loc[df['t'] == row['t'], 'MOKE'].values[0], axis=1)

# Prints de número de avalanchas según el filtro
aval1 = df_MOf.shape[0]
aval2 = df_MOf2.shape[0]
aval3 = df_MOa.shape[0]
print("Las avalanchas tras primer filtrado para el canal MOKE son", aval1)
print("Las avalanchas que duran más de una unidad de tiempo para el canal MOKE son", aval2)
print("El número de avalanchas independientes que duran más de t=1 para el canal MOKE es", aval3)
t_MOf = df_MOf['t'] # t minúscula porque representa los tiempos de ocurrencia de la avalancha

df_MOf = df_MOf[['MOKE', 'Corriente', 'dMOKE', 't', 'S']]
df_MOf

# %% Exploratory inspection of the data
# %%% MR and MOKE vs Voltage

# Later plots display data as a function of time instead of current.

plt.figure(figsize=(10, 6))
plt.plot(df['Corriente'], df['MOKE'], label='MOKE', color=palette[0], marker='o', markersize=3)
plt.plot(df['Corriente'], df['MR'], label='MR', color=palette[3], marker='x', markersize=3)

plt.xlabel('Corriente (A)')
plt.ylabel('Valor')
plt.title('MOKE y MR vs Corriente')

plt.legend()
plt.gca().invert_xaxis()

plt.grid(True)
plt.show()

# %%% MR and MOKE vs time

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # Second y axis

# All data

p1, = ax1.plot(df['t'], df['MOKE'], label='MOKE', color=palette[0], marker='o', markersize=2)
p2, = ax2.plot(df['t'], df['MR'], label='MR', color=palette[3], marker='x', markersize=2)

# Avalanches

p3, = ax2.plot(df_MRa['ti'], df_MRa['MR'], label='Avalanches MR', color=palette[2], marker='x', markersize=8, linestyle='None')
p4, = ax1.plot(df_MOf['t'], df_MOf['MOKE'], label='Avalanches MOKE', color=palette[2], marker='s', markerfacecolor='none', markersize=8, linestyle='None')

# Labels and format

ax1.set_xlabel('Time')
ax1.set_ylabel('MOKE', color=palette[0])
ax2.set_ylabel('MR', color=palette[3])
ax1.set_xlim(675, 800)
ax2.set_ylim(30,35)
ax1.set_title('MOKE and MR vs Time')

# Legend inside the plot

plots = [p1, p2, p3, p4]
labels = [p.get_label() for p in plots]
fig.legend(plots, labels, loc='upper left', bbox_to_anchor=(0.12, 0.88))
#plt.savefig('ejemplo_avalanchas.pdf')

plt.show()

# %%% Plot: voltage zoom (precursors)
# File: 190

fig, ax1 = plt.subplots(figsize=(5, 6))
#p1, = ax1.plot(df['Corriente'], df['MOKE'], label='MOKE', color=palette[0], marker='o', markersize=2)
p1, = ax1.plot(df['Corriente'], df['MR'], label='MR', color=palette[3], marker='x', markersize=1)

ax1.set_xlabel('V (V)', fontsize=25)
ax1.set_ylabel('MR ($\mu$V)', fontsize=25)
ax1.set_ylim(-35,25)
ax1.set_xlim(-0.03,-0.0275)
ax1.tick_params(axis='both',which='major',length=10, labelcolor='black', labelsize =22)
ax1.tick_params(axis='both',which='minor',length=4, labelcolor='black', labelsize=22)

plt.gca().invert_xaxis()
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
ax1.legend(plots, labels, loc='upper left', frameon=True,facecolor='white',
           edgecolor='lightgrey',framealpha=0.9,fancybox=True,shadow=False,  fontsize=22)

plt.savefig('ejemplo_avalanchas.pdf')
plt.tight_layout()
plt.show()

# %%%  Plot: voltage zoom (1 precursor)
# File: 190

fig, ax1 = plt.subplots(figsize=(8, 6))
#p1, = ax1.plot(df['Corriente'], df['MOKE'], label='MOKE', color=palette[0], marker='o', markersize=2)
p1, = ax1.plot(df['Corriente'], df['MR'], label='MR', color=palette[3], marker='x', markersize=1)

ax1.set_xlabel('V (V)', fontsize=25)
ax1.set_ylabel('MR ($\mu$V)', fontsize=25)
ax1.set_ylim(-13,-2)
ax1.set_xlim(-0.029,-0.0286)
ax1.tick_params(axis='both',which='major',length=10, labelcolor='black', labelsize =22)
ax1.tick_params(axis='both',which='minor',length=4, labelcolor='black', labelsize=22)

plt.gca().invert_xaxis()
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
ax1.legend(plots, labels, loc='upper left', frameon=True,facecolor='white',
           edgecolor='lightgrey',framealpha=0.9,fancybox=True,shadow=False,  fontsize=22)

plt.savefig('ejemplo_avalanchas1.pdf')
plt.tight_layout()
plt.show()

# %% Visual analysis
# %%% Noisy MOKE signal

plt.figure(figsize=(10, 6))

plt.plot(df['t'], df['MOKE']/2, label='MOKE/2', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMOKE'] / 10000, label='dMOKE/10000', color='m', marker='x', markersize=2)

# Fixed threshold for derivative

x = range(40,140)  
y = [5] * len(x)
plt.plot(x, y, label='Umbral del filtro', color='c', linestyle='--') 
y = [-5] * len(x)
plt.plot(x, y, color='b', linestyle='--')

# Graficar
plt.plot(x, y, color='c', linestyle='--')

plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Estudio del ruido de la señal MOKE')
plt.xlim(40,140)
plt.ylim(-10,10)

plt.legend()

plt.grid(True)
plt.show()

# La señal MOKE tiene más ruido que la señal MR así que hay que ser más cuidadosos al clasificar las avalanchas
# t=117 y t=68 no son avalanchas reales

# %%% Avalanche t=673 MOKE

plt.figure(figsize=(10, 6))

plt.plot(df['t'], df['MOKE']/5, label='MOKE/5', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMOKE']/10000, label='dMOKE/10000', color='m', marker='x', markersize=2)

# Umbral para la derivada
x = range(640,740)  
y = [5] * len(x)
plt.plot(x, y, label='Umbral del filtro', color='c', linestyle='--') 
y = [-5] * len(x)
plt.plot(x, y, color='c', linestyle='--')

plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Avalancha t=673 canal MOKE')
plt.xlim(668,680)
plt.ylim(-10,10)

plt.legend()

plt.grid(True)
plt.show()

# El principal problema es que la mayoría de las avalanchas reales en la señal MOKE tienen una duración de T=1
# Esto las hace especialmente difíciles de diferenciar del ruido
# La derivada y la señal están reescaladas para representarlas simultáneamente 

# %%% Avalanche t=671 MR

plt.figure(figsize=(10, 6))

plt.plot(df['t'], df['MR']/5, label='MR/2', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMR']/10000, label='dMR/10000', color='m', marker='x', markersize=2)

# Derivative threshold

x = range(640,740)  
#y = [5] * len(x)
#plt.plot(x, y, label='Umbral del filtro', color='b', linestyle='--')
y = [-5] * len(x)
plt.plot(x, y, color='c', linestyle='--', label='Umbral del filtro')

plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Avalancha t=671 canal MR')
plt.xlim(660,690)
plt.ylim(-50,10)

plt.legend()

plt.grid(True)
plt.show()

# En la señal MR en cambio la duración de la avalancha es mayor lo que permite identificarla mejor
# Podemos detectar antes la avalancha con la señal MR, en este caso 2 unidades de tiempo antes

# %%% Dynamic threshold MOKE

plt.figure(figsize=(10, 6))
plt.plot(df['t'], df['MOKE']/2, label='MOKE/2', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMOKE']/10000, label='dMOKE/10000', color='m', marker='x', markersize=2)

# Umbral variable de avalanchas
plt.plot(df['t'], -df['MOKE_umbral']/10000, label='Umbral variable', color='c', linestyle='--', marker='x', markersize=2)

plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Estudio del ruido de la señal MOKE')
plt.xlim(100,900)
plt.ylim(-40,30)

plt.legend()

plt.grid(True)
plt.show()

# Implementar un umbral variable mejora la identificación de las avalanchas
# Las avalanchas siguen teniendo una duración de T=1 en la mayoría de casos

# %%% Comparaing avalanches in both signals

# Consideramos que las avalanchas empiezan en t=400 y acaban en t=700
dfT = pd.DataFrame({ # DataFrame con los tiempos de avalancha de los dos canales
    't_MRa': t_MRa.astype(int),
    't_MOf': t_MOf.astype(int)
})

# Función para resaltar las celdas que cumplen la condición
def highlight_cells(val, col_name):
    if col_name == 't_MOf':
        # Si T_MOf es 1 o 2 unidades mayor que T_MRa
        if (val - 1) in t_MRa.values or (val - 2) in t_MRa.values:
            return 'background-color: red'
    elif col_name == 't_MRa':
        # Si T_MRa tiene un valor coincidente en T_MOf
        if (val + 1) in t_MOf.values or (val + 2) in t_MOf.values:
            return 'background-color: green'
    return ''

# Resaltar los tiempos que tienen un offset de t=1,2
styled_df = dfT.style.map(lambda v: highlight_cells(v, 't_MRa'), subset=['t_MRa']) \
                    .map(lambda v: highlight_cells(v, 't_MOf'), subset=['t_MOf'])

styled_df.format("{:.0f}")  # Esto asegura que se muestre como enteros

# %%% MOKE and MR avalanches

import matplotlib.patches as patches # Para agrupar los eventos MR en la leyenda

fig, ax = plt.subplots(figsize=(12, 4))

# Las avalanchas MR las dibujamos como rectángulos y las de MOKE como rectas para que se vea mejor si están dentro del rectángulo
for _, row in df_MRa.iterrows():
    rect = patches.Rectangle(
        (row['ti'], 0), row['tf'] - row['ti'], 1,
        color='red', alpha=0.5)
    ax.add_patch(rect)

# MOKE Avalanches

offset = 2 
for t in df_MOf['t']:
    ax.plot([t-offset, t-offset], [0, 1], color='blue', lw=1, label='Eventos MOKE' if t == df_MOf['t'].iloc[0] else "")

# Necessary for the legend

mr_patch = patches.Patch(color='red', alpha=0.5, label='Eventos MR')


ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_xlim(min(df_MRa['ti']) - 20, max(df_MRa['tf']) + 10)
ax.set_xlabel('Tiempo')  # Etiqueta del eje X
ax.set_title('Representación de avalanchas MR y MOKE')

# Legend

handles, labels = ax.get_legend_handles_labels()  # Detecta etiquetas automáticas
handles.append(mr_patch)  # Añade manualmente la etiqueta de MR
ax.legend(handles=handles, loc='upper right')  # Leyenda combinada

plt.grid()
plt.tight_layout()
plt.show()

# Mismo código de colores

# %%% Interval between t=535 and t=640

fig, ax = plt.subplots(figsize=(12, 4))

# Las avalanchas MR las dibujamos como rectángulos y las de MOKE como rectas para que se vea mejor si están dentro del rectángulo
for _, row in df_MRa.iterrows():
    rect = patches.Rectangle(
        (row['ti'], 0), row['tf'] - row['ti'], 1,
        color='red', alpha=0.5)
    ax.add_patch(rect)

# Avalanchas MOKE
offset = 1 
for t in df_MOf['t']:
    ax.plot([t-offset, t-offset], [0, 1], color='blue', lw=1, label='Eventos MOKE' if t == df_MOf['t'].iloc[0] else "")

# Necesario para que la leyenda se cree bien
mr_patch = patches.Patch(color='red', alpha=0.5, label='Eventos MR')


ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_xlim(535,600)
ax.set_xlabel('Tiempo')  # Etiqueta del eje X
ax.set_title('Representación de avalanchas MR y MOKE')

# Leyenda
handles, labels = ax.get_legend_handles_labels()  # Detecta etiquetas automáticas
handles.append(mr_patch)  # Añade manualmente  de MR
ax.legend(handles=handles, loc='upper right')  # Leyenda combinada

plt.grid()
plt.tight_layout()
plt.show()
# %%% Avalanche t=612 MOKE

plt.figure(figsize=(10, 6))

plt.plot(df['t'], df['MOKE']-10, label='MOKE - 10', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMOKE'] / 10000, label='dMOKE/10000', color='m', marker='x', markersize=2)

# Dynamic threshold for the derivative

plt.plot(df['t'], -df['MOKE_umbral']/10000, label='Umbral variable', color='c', linestyle='--', marker='x', markersize=2)

# Vertical line

y = range(-12,11)
x = [612]*len(y)
plt.plot(x, y, color='orange', label ='Inicio avalancha')
x = [615]*len(y)
plt.plot(x, y, color='orange', label ='Inicio avalancha')

plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Estudio de la avalancha t=612 en MOKE')
plt.xlim(600,630)
plt.ylim(-12,10)

plt.legend()

plt.grid(True)
plt.show()

# %% Avalanche plot

# %%% Function

def plot_avalancha(tiempo, ventana=20):
    '''

    Parameters
    ----------
    tiempo : TYPE
        Time where the avalanche occurs.
    ventana : TYPE, optional
        Amoun of points represented to the right and to the left of tiempo.
        The default is 20.

    Returns
    -------
    Creates the plot of MR and/or MOKE signal versus time.

    '''
    df_aux = df[(df['t'] >= tiempo - ventana) & (df['t'] <= tiempo + ventana)]
    plt.figure(figsize=(10, 6))
    
    # plt.plot(df['t'], df['MOKE'], label='MOKE', color='b', marker='o', markersize=2)
    plt.plot(df_aux['t'], df_aux['MR'], label='MR', color='r', marker='x', markersize=2)
    
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.title(f'Avalancha en t={int(tiempo)}')
    plt.xlim(tiempo-ventana,tiempo+ventana)
    plt.ylim(df_aux['MR'].min()-2, df_aux['MR'].max()+2)  
    legend = plt.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', borderpad=0.5, framealpha=1, fancybox=False)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    #plt.savefig(f'Avalancha-{int(tiempo)}.pdf') # para comprobar como quedan los gráficos
    plt.show()

# %%% Plot
plot_avalancha(tiempo=df_MRa.iloc[5]['ti']) # 7 avalanches, move index to visualize them

# %% MR precursors
# %%% Porcentual drop
# Porcentual drop in a window of T=20

pmatrix = np.zeros(len(df_MRa['ti']))
for i in range(len(df_MRa['ti'])):
    tiempo = df_MRa.iloc[i]['ti']  # Obtenemos el valor de 'ti'
    df_past = df[df['t'] <= tiempo]
    x_min = df_past['MR'].min()
    # Filtramos el DataFrame en torno a j
    df_aux = df[(df['t'] >= tiempo - 10) & (df['t'] <= tiempo)]
    df_aux = df_aux[['MR', 't']]
    min_aux = df_aux['MR'][tiempo] 
    # Cálculo del paval
    x = df_aux['MR'][tiempo] # Tomamos el instante donde empieza la avalancha, no el mínimo
    x_max = df_aux['MR'].max()
    print(x_max-x)
    paval = (x-x_min)/(x_max-x_min)
    #if (paval>0): paval = 0

    pmatrix[i] = paval

pmatrix
# %%% Detection
# First aproximation to precursor detection

pred = [] # Lista de 2-tuplas donde se guarda el tiempo y el resultado de la predicción
df_aval = pd.DataFrame(columns=["Inicio", "Final", "Result"])
for i in range(400,len(df['t'])):
    df_aux = df[(df['t'] >= i - 10) & (df['t'] <= i)]
    df_aux = df_aux[['MR', 't']]
    min_aux = df_aux['MR'][i]
    max_aux = df_aux['MR'].max()
    #paval = (max_aux - min_aux) * 100 / max_aux
    paval = max_aux-min_aux
    if (paval > 0.25): 
        pred.append(i)
i = len(pred) - 1  # Empezamos en el último índice y recorremos la lista hacia atrás
k = 1 # índice secundario para pasar de un grupo de predictores a otro
while i >= 0:
    if k ==1: # Guardamos el final
        final = pred[i]
    if pred[i] == pred[i-1] + 1:
        del pred[i]
        k = 2 # Recorremos el índice temporal hasta encontrar el inicio
    else:
        df_aux = df[(df['t'] >= pred[i] - 20) & (df['t'] <= final)]
        max_aux = df_aux['MR'].max()
        df_pred = df[(df['t'] > final) & (df['t'] <= final+5)]
        max_pred = df_pred['MR'].max()
        if max_pred > max_aux: result=1
        else: result=0
        new_entry = pd.DataFrame([{"Inicio": pred[i], "Final": final, "Result":result }]) # Nueva entrada
        #print(new_entry)
        df_aval = pd.concat([df_aval, new_entry], ignore_index=True) # Actualizamos el DataFrame
        k = 1 # Reinstauramos el índice para marcar un nuevo final
    i -= 1
print(len(df_aval))
print(df_aval['Result'].sum())


# %%% Precursors plot function

def plot_pred(df_aval, i, ventana=20): # le pasamos el dataframe y el índice de la detección que queremos visualizar
    '''

    Parameters
    ----------
    df_aval : TYPE
        Dataframe to visualize avalanches (MOKE or MR).
    i : TYPE
        DESCRIPTION.
    ventana : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.

    '''
    inicio = df_aval['Inicio'][i]
    final = df_aval['Final'][i]
    df_aux = df[(df['t'] >= inicio - ventana) & (df['t'] <= final + 10)]
    df_aux2 = df[(df['t'] >= inicio) & (df['t'] <= final)]
    df_aux3 = df[(df['t'] >= inicio-20) & (df['t'] <= final)]
    df_aux4 = df[(df['t'] > final) & (df['t'] <= final+5)]
    max_aux3 = df_aux3['MR'].max()
    max_pred = df_aux4['MR'].max()
    plt.figure(figsize=(10, 6))
    # plt.plot(df['t'], df['MOKE'], label='MOKE', color='b', marker='o', markersize=2)
    
    # Gráfico MR de toda la ventana
    plt.plot(df_aux['t'], df_aux['MR'], label='MR', color='r', marker='x', markersize=2)   
    
    # Zona de detección de la avalancha
    plt.plot(df_aux2['t'], df_aux2['MR'], label='Fase de detección', color='orange', marker='x', markersize=2)
    plt.axhline(y=max_aux3, color='orange', linestyle='--', label='Máximo de referencia')
    plt.fill_between(df_aux3['t'], df_aux3['MR'], max_aux3, color='orange', alpha=0.3)
    
    # Zona de evaluación del resultado de la detección
    plt.plot(df_aux4['t'], df_aux4['MR'], label='Fase de verificación', color='purple', marker='x', markersize=2)
    #plt.axhline(y=max_pred, color='purple', linestyle='--', label='Máximo en zona de evaluación')
    plt.fill_between(df_aux4['t'], df_aux4['MR'], max_aux3, color='purple', alpha=0.3)

    plt.xlabel('Tiempo')
    plt.ylabel('Canal MR')
    plt.title(f'Análisis de predicción de avalanchas: intervalo [t={int(inicio)}, t={int(final)}]')
    plt.xlim(inicio-ventana,final+10)
    plt.ylim(df_aux['MR'].min()-1, df_aux['MR'].max()+1)  
    legend = plt.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', borderpad=0.5, framealpha=1, fancybox=False)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    #plt.savefig(f'Predicción intervalo [t={int(inicio)}, t={int(final)}].pdf') # guarda los gráficos
    plt.show()


# %%% Plot
plot_pred(df_aval,1)
