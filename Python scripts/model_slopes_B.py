# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:53:30 2025

@author: iorti
"""

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

MRB = pd.read_csv("aval_MR_B_N.csv")
MOB = pd.read_csv("aval_MOKE_B_N.csv")

# %% Slope and magnetization between avalanches

rango = range(0,181) 
ajuste_MR = pd.DataFrame([[0,0,0,0]], columns=['M', 'm', 'b', 'Archivo'])

for i in rango:
    MRi = MRB[MRB['Archivo']==i]
    l = len(MRi)
    url = f'Datos3/hysteresis_deg_{i}.dat' 
    columnas = ['Corriente', 'MOKE', 'Hall', 'MR']
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['t']=range(len(df))
    df = df[df['Corriente']>=0.036290]
    maxM = max(df['MOKE'])
    df['M'] = (df['MOKE'] - df['MOKE'].min()) / (maxM - df['MOKE'].min())
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

ajuste_MR.to_csv('pendientes_B.csv', index=False)
        
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
d = ajuste_MR[ajuste_MR['M']<-25]
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
pendientes_B = pd.read_csv('pendientes_B.csv')

x = pendientes_B['M'].values
y = pendientes_B['m'].values

y_avg = np.mean(y)
y_std = np.std(y)
x_fit = np.linspace(min(x), max(x), 100)
print(f"Media: {y_avg:.4f}")
print(f"Desviación estándar: {y_std:.4f}")

plt.figure(figsize=(10,8))
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)

plt.scatter(x, y, color=palette[0], alpha=0.3, label='Data', zorder=2)
plt.plot(x_fit, [y_avg]*len(x_fit),'--', color=palette[1], zorder=2)

plt.fill_between(x_fit, y_avg - y_std, y_avg + y_std,color=palette[1],
    alpha=0.45,label=r'$m = \langle m \rangle \pm \mathrm{std}(m)$', zorder=2)
plt.xlabel(r'$\langle M \rangle$', fontsize=35)
plt.ylabel('m', fontsize=35)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(-0.2,0.8)
plt.tick_params(axis='x', which='major', length=12)
plt.tick_params(axis='x', which='minor', length=4)
plt.tick_params(axis='y', which='major', length=12)
plt.tick_params(axis='y', which='minor', length=4)
#plt.title('Slope between avalanches versus Magnetization')
plt.legend(frameon=True,facecolor='white',edgecolor='lightgrey',framealpha=0.9,
    loc='upper left',fancybox=True,shadow=False, fontsize=28)

plt.savefig('pendientes_B.pdf')
plt.show()

# %% Prediction with slope method

def crear_pred_m ():
    rango = range(0,182)
    columnas = ['Corriente', 'MOKE', 'Hall','MR']
    pred_m = pd.DataFrame([[0,0,0,0]], columns=['ti', 'tf','MR', 'Archivo'])
    for j in rango:
        if j % 10 == 0:
            print(j)
        predj = []
        url = f'Datos3/hysteresis_deg_{j}.dat'   
        df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
        df['t']=range(len(df))
        df = df[df['Corriente']>=0.036290]       
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
            
            m_ajuste = 0.0159 #0.0166
            #m_ajuste = p(M)
            if (m1+std_m1<m_ajuste):
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
    

# %%
def aval_plot(avMR, avMO, avP, i):
    url = f'Datos3/hysteresis_deg_{i}.dat' 
    columnas = ['Corriente', 'MOKE','Hall', 'MR']
    df = pd.read_csv(url, delim_whitespace=True, header =None, names = columnas)
    df['t']=range(len(df))     
    df = df[df['Corriente']>=0.036290]
    maxM = max(df['MOKE'])
    df['M'] = (df['MOKE'] - df['MOKE'].min()) / (maxM - df['MOKE'].min())
    avMR = avMR[avMR['Archivo']==i]
    avMO = avMO[avMO['Archivo']==i]
    avP = avP[avP['Archivo']==i]
    
    #df2 = df[(df['MOKE']<-22)&(df['MOKE']>-175)]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Second y axis

    # All data
    p1, = ax1.plot(df['t'], df['M'], label='MOKE', color=palette[0], marker='o', markersize=1)
    p2, = ax2.plot(df['t'], df['MR'], label='MR', color=palette[3], marker='x', markersize=1)
    #p6, = ax2.plot(df2['t'], df2['MR'], color=palette[4], marker='x', markersize=2, label='Critical region')

    # Avalanches
    p3, = ax2.plot(avMR['ti'], df['MR'][avMR['ti']], label='Avalanches MR', color=palette[2], marker='x', markersize=6, linestyle='None')
    p4, = ax1.plot(avMO['ti'], df['M'][avMO['ti']], label='Avalanches MOKE', color=palette[2], marker='s', markerfacecolor='none', markersize=6, linestyle='None')
    
    # Precursors
    p5, = ax2.plot(avP['tf'], avP['MR'], label='Precursors MR', color=palette[4], marker='s', markerfacecolor='none', markersize=6, linestyle='None')

    # Labels and format
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('MOKE', color=palette[0], fontsize=14)
    ax2.set_ylabel('MR', color=palette[3],  fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    #ax1.set_ylim(-200,200)

    # Legend below the figure
    plots = [p1, p2, p3, p4, p5]#, p6]
    labels = [p.get_label() for p in plots]
    ax1.legend(plots,labels,loc='lower right',frameon=True,facecolor='white',
    edgecolor='lightgrey',framealpha=0.9,fancybox=True,shadow=False,  fontsize=14)
    ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)
    plt.savefig(f'predictoresB_{i}.pdf')
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
         #   if (t + ventana >= taval) and (t <= tfaval):  # Verificas si ti_aval está en el rango [t+1, t+ventana]
        for taval in ti:  # Loop through each value in the list of avalanches for that file
            if t  <= taval <= t + ventana: 
                df.at[i, 'Result'] = 1
                df_aval.loc[(df_aval['ti'] == taval) & (df_aval['Archivo'] == file), 'M-Aux'] = 1
    A = df_aval['M-Aux'].sum()
    FP = l - df['Result'].sum()
    TE = 1-FP/l # Success rate (TE='Tasa de éxito')
    return ([A,FP, TE, l])

# %% Precursor metrics - data set A
#pred_m_B = crear_pred_m()
#pred_m_B.to_csv('pred_m_B.csv', index=False)
pred_m_B = pd.read_csv('pred_m_B.csv')
metricas_m = verificar_prediccion(pred_m_B, MRB)
print(metricas_m)

# %%
aval_plot(MRB, MOB, pred_m_B, 52)