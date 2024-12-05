

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import pandas as pd




""" Defino mis funciones """
    
def f_conc_salida(t, caudal_entrada, caudal_salida, conc_entrada, conc_inicial, volumen_inicial):   
        
    if caudal_entrada == caudal_salida:     
        caudal_entrada = caudal_entrada + 0.00001
    
    return (conc_entrada / (conc_entrada - conc_inicial) - (1 + (caudal_entrada - caudal_salida) * t / volumen_inicial) ** (caudal_entrada / (caudal_salida - caudal_entrada))) * (conc_entrada - conc_inicial)

    
def f_volumen(t, volumen_anterior, caudal_entrada, caudal_salida):
    return volumen_anterior + (caudal_entrada - caudal_salida) * t




def generarSeriesOriginales(intervalos_tiempo, lista_caudal_entrada, lista_caudal_salida, lista_conc_entrada, conc_inicial, volumen_inicial):
    
    cant_series = len(lista_caudal_entrada)
    
    lista_conc_salida = cant_series * [np.zeros(intervalos_tiempo)]
    lista_volumenes = cant_series * [np.zeros(intervalos_tiempo)]
    lista_tiempos = cant_series * [np.arange(intervalos_tiempo)]
    
    for i in range(cant_series):
        
        # Defino mis Vars
        caudales_entrada_serie = lista_caudal_entrada[i]
        caudales_salida_serie = lista_caudal_salida[i]
        conc_entrada_serie = lista_conc_entrada[i]
        tiempos_serie = lista_tiempos[i]
        
        
        # Computo Vols
        
        for j in range(len(tiempos_serie)):
            if(j==0):
                lista_volumenes[i][j] = volumen_inicial
            else:
                lista_volumenes[i][j] = f_volumen(tiempos_serie[j]-tiempos_serie[j-1], lista_volumenes[i][j-1], caudales_entrada_serie[j-1], caudales_salida_serie[j-1])
        
        
        # Computo Conc
        
        t=0
        for j in range(len(tiempos_serie)):
            
            # Me fijo Cambio de Condiciones de Operación
            
            if(j!=0 and ( caudales_entrada_serie[j-1] != caudales_entrada_serie[j] or caudales_salida_serie[j-1] != caudales_salida_serie[j] or conc_entrada_serie[j-1] != conc_entrada_serie[j])):
                
                volumen_inicial = lista_volumenes[i][j]
                t = 0
            
            
            # Me Guardo la conc salida
            
            lista_conc_salida[i][j] = f_conc_salida(t, caudales_entrada_serie[j], caudales_salida_serie[j], conc_entrada_serie[j], conc_inicial, volumen_inicial)
            t = t + 1
        
        
        return (lista_conc_salida, lista_volumenes, lista_tiempos)

    
def adimensionalizarAsumiendoConstancia(lista_tiempos, lista_caudal_entrada, lista_conc_entrada, lista_conc_salida):

        # Xa = 1-(1+Ta*R)**-R^-1
        # R = (Qe-Qs)/Qe
        # Ta = (Qe/Vi) * t         #Dq
        # Xa = Xs/Xe
        # Va= Vi*(1 + R*Ta)
        # Cap otra opcion
        
    lista_conc_salida_adim = len(lista_conc_salida) * [None]
    lista_tiempos_adim = len(lista_conc_salida) * [None]
    for i in range(len(lista_conc_salida)):
        lista_conc_salida_adim[i] = lista_conc_salida[i] / lista_conc_entrada[i][0]
        lista_tiempos_adim[i] = lista_tiempos[i] * lista_caudal_entrada[i][0]

    return(lista_conc_salida_adim, lista_tiempos_adim)


def listasRedimensionalizadasAEnteros(lista_tiempos_adim, vec_caudal_entrada_redim, vec_conc_entrada_redim, volumen_inicial,  lista_conc_salida_adim, interpolado=True):
    
    # Defino Tamaños
    cant_cond_oper = len(vec_conc_entrada_redim) * len(vec_caudal_entrada_redim)
    cant_series = cant_cond_oper * len(lista_conc_salida_adim)
    
    lista_series_redim = cant_series*[None]
    lista_t_redim = cant_series*[None]
    lista_conc_entrada = cant_series*[None]
    lista_caudal_entrada = cant_series*[None]

    
    numero_serie=0
    for i in range(len(lista_conc_salida_adim)):
        
        serie = lista_conc_salida_adim[i]
        t_adim =  lista_tiempos_adim[i]

        for conc_entrada in vec_conc_entrada_redim:
            for caudal_entrada in vec_caudal_entrada_redim:
                
                # Redimensionalizacion
                
                serie_redim = serie * conc_entrada
                t_redim = t_adim * volumen_inicial / caudal_entrada
                
                
                # Interpolo cuadrático
                if(interpolado==True):
                    f2 = interpolate.interp1d(t_redim , serie_redim, kind='quadratic')
                
                    t_redim_entero = np.arange(int(t_redim[-1]))
                    serie_redim_entero = f2(t_redim_entero)
                    
                    
                    lista_series_redim[numero_serie] = serie_redim_entero.reshape((-1))
                    lista_t_redim[numero_serie] = t_redim_entero.reshape((-1))
                    

                else:
                    lista_series_redim[numero_serie] = serie_redim.reshape((-1))
                    lista_t_redim[numero_serie] = t_redim.reshape((-1))
                
                
                
                lista_conc_entrada[numero_serie] = np.full(shape=lista_series_redim[numero_serie].shape[0], fill_value=conc_entrada) 
                lista_caudal_entrada[numero_serie] = np.full(shape=lista_series_redim[numero_serie].shape[0], fill_value=caudal_entrada)
                
                
                

                
                
                # Actualizo
                numero_serie = numero_serie + 1
                
    return((lista_series_redim, lista_t_redim, lista_conc_entrada, lista_caudal_entrada))


def pasarAMatriz(lista_de_arrays):
    
    cantidad_arrays = len(lista_de_arrays)
    max_len_array = 0
    for i in range(cantidad_arrays):
        if(len(lista_de_arrays[i]) > max_len_array):
            max_len_array = len(lista_de_arrays[i])
            
    matriz = np.full(shape=(max_len_array, cantidad_arrays), fill_value=-1, dtype = np.float64)
    
    for j in range(cantidad_arrays): 
        array = lista_de_arrays[j] + 1
        array = np.pad(array, (0, max_len_array - len(array)), 'constant')
        matriz[:, j] =  matriz[:,j] + array
                                            
    return(matriz)


def producirDatos(cant_condiciones, min_conc_entrada, max_conc_entrada, min_caud_entrada, max_caud_entrada, interp=False):
    intervalos_tiempo = 100 


    lista_caudal_entrada = [
                            np.full(shape=intervalos_tiempo, fill_value=1),
                            np.full(shape=intervalos_tiempo, fill_value=2),
                            np.full(shape=intervalos_tiempo, fill_value=3)
                          ]


    lista_caudal_salida = [
                            np.full(shape=intervalos_tiempo, fill_value=0.5),
                            np.full(shape=intervalos_tiempo, fill_value=2),
                            np.full(shape=intervalos_tiempo, fill_value=2.5)
                        ]

    lista_conc_entrada = [
                            np.full(shape=intervalos_tiempo, fill_value=2),
                            np.full(shape=intervalos_tiempo, fill_value=2),
                            np.full(shape=intervalos_tiempo, fill_value=3)
                        ]


    conc_inicial = 0                    # en tanque
    volumen_inicial = 30                # en tanque
    
    
    
    lista_conc_salida, lista_volumenes, lista_tiempos = generarSeriesOriginales(intervalos_tiempo, lista_caudal_entrada, lista_caudal_salida, lista_conc_entrada, conc_inicial, volumen_inicial)




    lista_conc_salida_adim, lista_tiempos_adim = adimensionalizarAsumiendoConstancia(lista_tiempos, lista_caudal_entrada, lista_conc_entrada, lista_conc_salida)

    

    vec_conc_entrada_redim = np.linspace(min_conc_entrada, max_conc_entrada, cant_condiciones)
    vec_caudal_entrada_redim = np.linspace(min_caud_entrada, max_caud_entrada, cant_condiciones)   


    lista_series_redim, lista_t_redim, lista_conc_entrada, lista_caudal_entrada = listasRedimensionalizadasAEnteros(lista_tiempos_adim, vec_caudal_entrada_redim, vec_conc_entrada_redim, volumen_inicial,  lista_conc_salida_adim, interpolado=interp)
    
    
    

    return((lista_series_redim, lista_t_redim, lista_conc_entrada, lista_caudal_entrada))

lista_series_redim, lista_t_redim, lista_conc_entrada, lista_caudal_entrada = producirDatos(cant_condiciones= 20, min_conc_entrada= 2, max_conc_entrada= 5, min_caud_entrada=5, max_caud_entrada=12, interp=True)


#for i in range(270):
#    plt.plot(lista_series_redim[i*10])




