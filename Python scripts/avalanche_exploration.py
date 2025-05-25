# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import powerlaw
import scienceplots # gráficos con estilo unificado y formato latex
plt.style.use('science')
palette = sns.color_palette("muted")


# %% Inicio
columnas = ['Corriente', 'MOKE', 'MR']
url = 'Datos/hysteresis_deg_0.dat'
df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)

df['dMOKE'] = df['MOKE'].diff(-1) / df['Corriente'].diff(-1)
df['dMR'] = df['MR'].diff(-1) / df['Corriente'].diff(-1)
l = len(df)
df['dMR'][l-1]=0
df['dMOKE'][l-1]=0
df['t']=range(len(df))

# %% Canal MR

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
# %% Canal MOKE con ventana a la izquierda
# Subiendo de 2.8 a 3 perdemos la avalancha de t=497
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

# %% Gráfico de MR y MOKE vs Corriente
# Posteriormente se realizan los gráficos en función del tiempo
plt.figure(figsize=(10, 6))
plt.plot(df['Corriente'], df['MOKE'], label='MOKE', color='b', marker='o', markersize=3)
plt.plot(df['Corriente'], df['MR'], label='MR', color='r', marker='x', markersize=3)

plt.xlabel('Corriente (A)')
plt.ylabel('Valor')
plt.title('MOKE y MR vs Corriente')

plt.legend()

plt.grid(True)
plt.show()

# %% Gráfico de MR y MOKE vs tiempo
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # Segundo eje y

# Datos totales
p1, = ax1.plot(df['t'], df['MOKE'], label='MOKE', color=palette[0], marker='o', markersize=2)
p2, = ax2.plot(df['t'], df['MR'], label='MR', color=palette[3], marker='x', markersize=2)

# Avalanchas
p3, = ax2.plot(df_MRa['ti'], df_MRa['MR'], label='Avalanches MR', color=palette[2], marker='x', markersize=8, linestyle='None')
p4, = ax1.plot(df_MOf['t'], df_MOf['MOKE'], label='Avalanches MOKE', color=palette[2], marker='s', markerfacecolor='none', markersize=8, linestyle='None')

# Etiquetas y formato
ax1.set_xlabel('Time')
ax1.set_ylabel('MOKE', color=palette[0])
ax2.set_ylabel('MR', color=palette[3])
ax1.set_xlim(675, 800)
ax2.set_ylim(30,35)
ax1.set_title('MOKE and MR vs Time')

# Leyenda dentro del gráfico
plots = [p1, p2, p3, p4]
labels = [p.get_label() for p in plots]
fig.legend(plots, labels, loc='upper left', bbox_to_anchor=(0.12, 0.88))
#plt.savefig('ejemplo_avalanchas.pdf')

plt.show()

# %% Gráfico Ruido señal MOKE
plt.figure(figsize=(10, 6))

plt.plot(df['t'], df['MOKE']/2, label='MOKE/2', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMOKE'] / 10000, label='dMOKE/10000', color='m', marker='x', markersize=2)

# Umbral fijo para la derivada
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

# %% Gráfico avalancha t=673 canal MOKE
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

# %% Gráfico avalancha t=671 canal MR
plt.figure(figsize=(10, 6))

plt.plot(df['t'], df['MR']/5, label='MR/2', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMR']/10000, label='dMR/10000', color='m', marker='x', markersize=2)

# Umbral para la derivada
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

# %% Gráfico umbral variable señal MOKE
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

# %% Comparación de los tiempos
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

# %% Gráfico comparativa de avalanchas para MOKE y MR

import matplotlib.patches as patches # Para agrupar los eventos MR en la leyenda

fig, ax = plt.subplots(figsize=(12, 4))

# Las avalanchas MR las dibujamos como rectángulos y las de MOKE como rectas para que se vea mejor si están dentro del rectángulo
for _, row in df_MRa.iterrows():
    rect = patches.Rectangle(
        (row['ti'], 0), row['tf'] - row['ti'], 1,
        color='red', alpha=0.5)
    ax.add_patch(rect)

# Avalanchas MOKE
offset = 2 
for t in df_MOf['t']:
    ax.plot([t-offset, t-offset], [0, 1], color='blue', lw=1, label='Eventos MOKE' if t == df_MOf['t'].iloc[0] else "")

# Necesario para que la leyenda se cree bien
mr_patch = patches.Patch(color='red', alpha=0.5, label='Eventos MR')


ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_xlim(min(df_MRa['ti']) - 20, max(df_MRa['tf']) + 10)
ax.set_xlabel('Tiempo')  # Etiqueta del eje X
ax.set_title('Representación de avalanchas MR y MOKE')

# Leyenda
handles, labels = ax.get_legend_handles_labels()  # Detecta etiquetas automáticas
handles.append(mr_patch)  # Añade manualmente la etiqueta de MR
ax.legend(handles=handles, loc='upper right')  # Leyenda combinada

plt.grid()
plt.tight_layout()
plt.show()

# Mismo código de colores

# %% Detalle del intervalo entre t=535 y t=640

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
plt.show(
# %% Estudio avalancha t=612 MOKE
plt.figure(figsize=(10, 6))

plt.plot(df['t'], df['MOKE']-10, label='MOKE - 10', color='b', marker='o', markersize=2)
plt.plot(df['t'], df['dMOKE'] / 10000, label='dMOKE/10000', color='m', marker='x', markersize=2)

# Umbral variable para la derivada
plt.plot(df['t'], -df['MOKE_umbral']/10000, label='Umbral variable', color='c', linestyle='--', marker='x', markersize=2)

#Linea vertical
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

# Las avalanchas que aparecían antes en t=615 y t=616 no aparencen por poco con el nuevo filtro
# Viendo la evolución del canal MOKE se tratan de avalanchas pequeñas
# En este caso ambas avalanchas podrían formar una única avalancha en el canal MR

# %% power law
def plot_power_law(data, titulo,pdf='prueba', nb=10000):
    # Límites
    start, end = min(data), max(data)

    # Histograma en escala logarítmica
    log_bins = np.histogram_bin_edges(np.log(data), bins='auto')  # Bins en escala log
    bins = np.exp(log_bins)  # Volver a escala original
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    # Calcular medias de los bins (centros)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Estimación del exponente α por máxima verosimilitud
    xmin = min(data)
    alpha_mle = 1 + len(data) / np.sum(np.log(data / xmin))

    # Bootstrap para estimar intervalos de confianza
    densb = np.zeros((nb, len(bin_mids))) # Inicializa, crea una matriz de ceros

    for b in range(nb):
        sample_b = np.random.choice(data, size=len(data), replace=True)  # Resampling con reemplazo
        sample_b = np.clip(sample_b, bins[0], bins[-1])  # Ajuste de valores fuera del rango
        hist_b, _ = np.histogram(sample_b, bins=bin_edges, density=True)
        densb[b, :] = hist_b

    # Calcular intervalos de confianza del 95%
    ci_lower = np.percentile(densb, 2.5, axis=0)
    ci_upper = np.percentile(densb, 97.5, axis=0)
    
    # Graficar la ley de potencias ajustada con alpha_MLE
    x_fit = np.linspace(start, end, 100)
    y_fit = (alpha_mle - 1) * (xmin ** (alpha_mle - 1)) * x_fit ** (-alpha_mle)
    plt.figure(figsize=(8, 6))
    plt.plot(x_fit, y_fit, 'b--', label=f'Ajuste Ley de Potencias ($\\alpha={alpha_mle:.2f}$)')

    # Graficar la distribución empírica con incertidumbre bootstrap
    plt.plot(bin_mids, hist, color='red', label='Distribución empírica', marker='o', markersize=4)

    # Agregar barras de error con los intervalos de confianza
    plt.errorbar(bin_mids, hist, yerr=[hist - ci_lower, ci_upper - hist], color='red', capsize=3)

    # Configuración de ejes y etiquetas
    plt.xscale('log')
    plt.yscale('log')
    plt.style.use('science')
    plt.xlim(start, end)
    plt.xlabel('Log(Tamaño de la avalancha)')
    plt.ylabel('Log(Densidad de probabilidad)')
    plt.title(f'Densidad de probabilidad vs Tamaño de la avalancha (S) - {titulo}')
    legend = plt.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', borderpad=0.5, framealpha=1, fancybox=False)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black")    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.savefig(f'{pdf}.pdf') # para comprobar como quedan los gráficos
    plt.show()


# %% power law MR
data = np.array(df_MRa['S'])
plot_power_law(data, nb=10000, titulo='Canal MR', pdf='DistMR')

# %% power law MOKE
data = np.array(df_MOf['S'])
plot_power_law(data, nb=10000, titulo='Canal MOKE', pdf='DistMOKE')


# %% Survival
def S_power_law(data, titulo, pdf='prueba'):
    xmin = min(data)     # Umbral mínimo
    xmax = max(data)
    n = len(data)      # Número de datos
    alpha_mle = 1 + n / np.sum(np.log(data / xmin))

    # Empírica
    data_sorted = np.sort(data)
    S_empirical = 1 - np.linspace(1/n, 1, n)
    
    # Teórica
    x_fit = np.linspace(xmin, xmax, 100)
    y_fit = (x_fit/xmin)**(-(alpha_mle-1))
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_fit, y_fit, 'b--', label=f'Teórica ($\\alpha={alpha_mle:.2f}$)')
    plt.plot(data_sorted, S_empirical, 'o', linestyle='-', markersize=3, color='red', label="Empírica (datos)", zorder=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Tamaño de la avalancha (S)")
    plt.ylabel("$\\mathbb{P}(X)>x$")
    plt.title(f"Función de Supervivencia (Size) - {titulo}")
    plt.style.use('science')
    
    legend = plt.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', borderpad=0.5, framealpha=1, fancybox=False)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black") 
    plt.grid(True,which='both', linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
    
    plt.savefig(f'{pdf}.pdf') # para comprobar como quedan los gráficos
    plt.show()


# %% SUrvival MOKE
data = np.array(df_MOf['S'])
S_power_law(data, titulo='Canal MOKE', pdf='S_MOKE')

# %% Survival MR
data = np.array(df_MRa['S'])
S_power_law(data, titulo='Canal MR', pdf='S_MR')


# %% Plot avalancha
# Gráfico de MR y MOKE vs tiempo
def plot_avalancha(tiempo, ventana=20):
    df_aux = df[(df['t'] >= tiempo - ventana) & (df['t'] <= tiempo + ventana)]
    plt.figure(figsize=(10, 6))
    # Datos totales
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

plot_avalancha(tiempo=df_MRa.iloc[5]['ti']) # hay 7 avalanchas, cambiar índice para visualizarlas

# %% Variación previa avalanchas MR
# Variación previa a la avalancha del canal MR
# Miramos el descenso porcentual respecto una ventana de T=20
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
# %% Predictores
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

# %%
plot_avalancha(tiempo=565, ventana=20)


# %% Gráfico predictor
# Función para graficar las detección y verificación de avalanchas
def plot_pred(df_aval, i, ventana=20): # le pasamos el dataframe y el índice de la detección que queremos visualizar
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


# %%
plot_pred(df_aval,1)

# %%
