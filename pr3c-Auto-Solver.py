####################################
# LIBRERIAS 
####################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import time
import multiprocessing as mp
from itertools import product
import os

#######################
#####  CONSTANTES  ####
#######################
days = 24 * 3600                    # segundos/día
G = 6.6742e-20                      # km^3/kg/s^2
rmoon = 1737                        # radio de la Luna km
rearth = 6378                       # radio de la Tierra km
r12 = 384400                        # distancia Tierra-Luna km

m1 = 5974e21                        # masa Tierra kg
m2 = 7348e19                        # masa Luna kg
M = m1 + m2
pi_1 = m1 / M
pi_2 = m2 / M

mu1 = 398600                        # parámetro gravitacional Tierra km^3/s^2
mu2 = 4903.02                       # parámetro gravitacional Luna km^3/s^2
mu = mu1 + mu2

C1 = -1.67339716
C2 = -1.66490460
C3 = -1.58091856
C_12 = 0.5 * (C1 + C2)
C_13 = 0.5 * (C1 + C3)

W = np.sqrt(mu / r12**3)            # velocidad angular rad/s

x1 = -pi_2 * r12                    # posición x de la Tierra en el sistema rotante
x2 = pi_1 * r12                     # posición x de la Luna

L1 = 321710                         # distancia L1 km

# Parámetros de propulsión y otros
n = 4#1#4                   #busek 1.1mN
F = 0.00000045 #0.0000011 #0.0000004                      # empuje
T_val = F * n                       # empuje en kN
m_motor = 1.875
m_cap = 2.0
tol = 1e-12

#################################
# Funciones de propagacion
##################################
def rates(t, f):
    """Fase con empuje - primera trayectoria"""
    x, y, vx, vy, m = f
    r1_val = np.linalg.norm([x + pi_2 * r12, y])
    r2_val = np.linalg.norm([x - pi_1 * r12, y])
    v_val = np.linalg.norm([vx, vy])
    ax = 2 * W * vy + W**2 * x - mu1 * (x - x1) / (r1_val**3) - mu2 * (x - x2) / (r2_val**3) + (T_val / m) * (vx / v_val)
    ay = -2 * W * vx + W**2 * y - (mu1/(r1_val**3) + mu2/(r2_val**3)) * y + (T_val / m) * (vy / v_val)
    g0 = 9.807
    Isp = 1650
    mdot = -T_val * 1000 / (g0 * Isp)
    return [vx, vy, ax, ay, mdot]

def rates0(t, f):
    """Fase de coasting - sin empuje"""
    x, y, vx, vy, m = f
    r1_val = np.linalg.norm([x + pi_2 * r12, y])
    r2_val = np.linalg.norm([x - pi_1 * r12, y])
    ax = 2 * W * vy + W**2 * x - mu1 * (x - x1) / (r1_val**3) - mu2 * (x - x2) / (r2_val**3)
    ay = -2 * W * vx + W**2 * y - (mu1/(r1_val**3) + mu2/(r2_val**3)) * y
    return [vx, vy, ax, ay, 0]

def rates_1(t, f):
    """Fase de frenado - empuje negativo"""
    x, y, vx, vy, m = f
    r1_val = np.linalg.norm([x + pi_2 * r12, y])
    r2_val = np.linalg.norm([x - pi_1 * r12, y])
    v_val = np.linalg.norm([vx, vy])
    T_neg = -F * n  # empuje invertido (frena)
    ax = 2 * W * vy + W**2 * x - mu1 * (x - x1) / (r1_val**3) - mu2 * (x - x2) / (r2_val**3) + (T_neg / m) * (vx / v_val)
    ay = -2 * W * vx + W**2 * y - (mu1/(r1_val**3) + mu2/(r2_val**3)) * y + (T_neg / m) * (vy / v_val)
    g0 = 9.807
    Isp = 1650
    mdot = -abs(T_neg) * 1000 / (g0 * Isp)
    return [vx, vy, ax, ay, mdot]

####################################
# Funciones de eventos
####################################
# En esta versión, definiremos el evento de Jacobi para la fase 1 como una función local
# que usa un umbral variable (jacobi_threshold)
Y_WINDOW_L1 = 55000.0  # km

def lagranian1(t, state, phiS0_rad):
    """
    Dispara cuando la trayectoria cruza la línea x = x_L1 garganta de L1,
    pero solo si |y| <= Y_WINDOW_L1. Fuera de esa franja en Y, la función
    se mantiene lejos de cero para evitar raíces espurias.
    """
    x_val, y_val, *_ = state

    x_L1 = x1 + L1
    s = x_val - x_L1  # distancia horizontal a la línea de L1

    if abs(y_val) <= Y_WINDOW_L1:
        # Dentro de la franja vertical: raíz cuando x cruza x_L1
        return s
    eps = 1e-6  # tolerancia para evitar problemas
    margin = (abs(y_val) - Y_WINDOW_L1) + eps
    return (1.0 if s >= 0.0 else -1.0) * margin

lagranian1.terminal = True
lagranian1.direction = 0



def jacobiC1(t, y):
    """Evento: se dispara cuando la constante de Jacobi alcanza C1 en la fase de frenado"""
    x_val, y_val, vx, vy, _ = y
    v_val = np.linalg.norm([vx, vy])
    r1_val = np.linalg.norm([x_val + pi_2 * r12, y_val])
    r2_val = np.linalg.norm([x_val - pi_1 * r12, y_val])
    return 0.5*v_val**2 - 0.5*W**2*(x_val**2+y_val**2) - mu1/r1_val - mu2/r2_val - C1
jacobiC1.terminal = True
jacobiC1.direction = 0

def circular(t, y):
    """Evento de circularización: cuando el producto punto entre la posición relativa a la Luna y la velocidad es cero"""
    x_val, y_val, vx, vy, _ = y
    r2_vec = np.array([x_val - pi_1 * r12, y_val])
    v_vec = np.array([vx, vy])
    return np.dot(r2_vec, v_vec)
circular.terminal = True
circular.direction = 0

def collision_event(t, y):
    """Evento: se dispara cuando la distancia al centro lunar es igual al radio lunar"""
    x_val, y_val, _, _, _ = y
    d_moon = np.linalg.norm([x_val - x2, y_val])
    return d_moon - rmoon
collision_event.terminal = True
collision_event.direction = 0

def capture_event(t, y):
    """Evento: se dispara cuando la energía relativa a la Luna se vuelve negativa"""
    x_val, y_val, vx, vy, _ = y
    r_rel = np.linalg.norm([x_val - x2, y_val])
    speed = np.linalg.norm([vx, vy])
    E = 0.5 * speed**2 - mu2 / r_rel
    return E
capture_event.terminal = True
capture_event.direction = -1

####################################
# Función auxiliar para trazar círculos (Tierra, Luna)
####################################
def circle(xc, yc, radius, num_points=361):
    theta = np.deg2rad(np.linspace(0, 360, num_points))
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    return x, y

def save_result(result):
    df = pd.DataFrame([result])
    filename = 'resultados_trayectoriav003b.csv'
    if not os.path.exists(filename):
        df.to_csv(filename, index= False)
    else:
        df.to_csv(filename,mode='a',header=False,index=False)


####################################
# Función principal de la simulación que recibe un trío de parámetros:
# (phi, jacobi_threshold, d0)
####################################
def trayectoria(params):
    phi, jacobi_threshold, d0 = params
    start_time = time.time()
    # Parámetros iniciales
    # d0 es la altitud inicial km pasada como parámetro
    h_apogee = 37000    # Altitud del apogeo km
    h_perigee = 1200     # Altitud del perigeo km
    r_apogee = rearth + h_apogee  # Radio en el apogeo
    r_perigee = rearth + h_perigee  # Radio en el perigeo

    # Cálculo del semieje mayor y la excentricidad de la órbita elíptica
    a = (r_apogee + r_perigee) / 2
    e = (r_apogee - r_perigee) / (r_apogee + r_perigee)

    # Velocidad en el apogeo para la órbita GTO (usando la ecuación de vis-viva)
    v0 = np.sqrt(mu1 * (1 - e) / r_apogee) - W * r_apogee

    # Otros parámetros
    gamma = 0    # Ángulo de vuelo inicial (grados)
    t0 = 0
    tf = days * 360 * 4   # Tiempo máximo de integración (s)
    r0 = r_apogee         # Radio inicial igual al apogeo

    # Selección de ángulo (se mantiene el mismo valor de salida)
    #phi = 5  # en grados

    # Condiciones iniciales en el sistema (se utiliza el apogeo como punto de partida)
    phi_rad = np.deg2rad(phi)
    gamma_rad = np.deg2rad(gamma)
    x0 = r0 * np.cos(phi_rad) + x1
    y0 = r0 * np.sin(phi_rad)
    vx0 = v0 * (np.sin(gamma_rad) * np.cos(phi_rad) - np.cos(gamma_rad) * np.sin(phi_rad))
    vy0 = v0 * (np.sin(gamma_rad) * np.sin(phi_rad) + np.cos(gamma_rad) * np.cos(phi_rad))
    m0_val = 12
    f0 = [x0, y0, vx0, vy0, m0_val]

    # Definir evento de Jacobi para fase 1 usando el umbral variable
    def jacobiC_local(t, y):
        x_val, y_val, vx, vy, _ = y
        v_val = np.linalg.norm([vx, vy])
        r1_val = np.linalg.norm([x_val + pi_2 * r12, y_val])
        r2_val = np.linalg.norm([x_val - pi_1 * r12, y_val])
        return 0.5*v_val**2 - 0.5*W**2*(x_val**2+y_val**2) - mu1/r1_val - mu2/r2_val - jacobi_threshold
    jacobiC_local.terminal = True
    jacobiC_local.direction = 0

    exito = True
    capture_success = False
    # Fase 1: Empuje (con evento jacobiC_local)
    sol1 = solve_ivp(rates, [t0, tf], f0, method='RK45', events=jacobiC_local,
                     rtol=1e-9, atol=tol, max_step=750)
    print('Fase1 phi ',phi,' tfinal ', sol1.t[-1])
    if sol1.t_events[0].size > 0:
        print(f"[Fase 1] Evento jacobiC_local disparado en t = {sol1.t_events[0]}")
        print(f"[Fase 1] Estado en evento: {sol1.y_events[0]}")
    else:
        print("[Fase 1] No se disparó el evento jacobiC_local")
    f1_final = sol1.y[:, -1]

    # Fase 2: Coasting
    t_phase2 = [sol1.t[-1], sol1.t[-1] + days * 650]
    sol2 = solve_ivp(rates0, t_phase2, f1_final, method='RK45', events=lagranian1,
                     rtol=1e-9, atol=tol, max_step=400 , dense_output=True)
    print(f"[Fase 2] phi={phi}: t_final = {sol2.t[-1]} s")
    #f2_final = sol2.y[:, -1]
    if sol2.t_events[0].size > 0:
        print('SE disparo Lagrian1')
        f2_final = sol2.y[:, -1]
    else:
        print(f"[Fase 2] Evento lagrian1 NO se disparó")
        exito = False
        tiempo_total = sol2.t[-1]
        masa_final = sol2.y[4,-1]
        detalles = {
        'phi': phi,
        'jacobi_threshold': jacobi_threshold,  # Agregado
        'd0': d0,    
        'tiempo_total': tiempo_total,
        'masa_final': masa_final,
        'exito': exito,
        'SOI': False,
        'time_exec': time.time() - start_time
        }
        return detalles

    # Fase 3: Frenado para inserción lunar
    t_phase3 = [sol2.t[-1], sol2.t[-1] + days * 180]
    sol3 = solve_ivp(rates_1, t_phase3, f2_final, method='RK45',
                     events=[jacobiC1, collision_event, capture_event],
                     rtol=1e-9, atol=tol, max_step=200, dense_output=True)
    f3_final = sol3.y[:, -1]

    d_phase2 = np.sqrt((sol2.y[0] - x2)**2 + (sol2.y[1])**2)
    d_phase3 = np.sqrt((sol3.y[0] - x2)**2 + (sol3.y[1])**2)
    min_distance = min(np.min(d_phase2), np.min(d_phase3))
    SOI_flag = (min_distance <= 50000)
    print("Distancia mínima a la Luna:", min_distance, "km, SOI =", SOI_flag)

    if len(sol3.t_events[1]) > 0:
        if len(sol3.t_events[1]) > 0 : print('Choco la LUNA phi -',phi,)
        exito = False
        tiempo_total = sol2.t[-1]
        masa_final = sol2.y[4,-1]
    else:
        t_phase3 = sol3.t
        x_phase3 = sol3.y[0]
        y_phase3 = sol3.y[1]
        vx_phase3 = sol3.y[2]
        vy_phase3 = sol3.y[3]

        # Calcular la distancia relativa a la Luna
        r_rel_phase3 = np.sqrt((x_phase3 - x2)**2 + (y_phase3)**2)

        # Calcular la energía de captura en cada instante
        E_capture = 0.5 * (vx_phase3**2 + vy_phase3**2) - mu2 / r_rel_phase3

        # Verificar si en todo el intervalo se cumple E < 0
        if np.all(E_capture < 0):
            print("La condición de captura (E < 0) se cumple - phi", phi)
            capture_success = True
            exito = True
            tiempo_total = sol3.t[-1]
            masa_final = sol3.y[4, -1]
        else:
            print("La condición de captura NO - ", phi)
            capture_success = False
            exito = False
            tiempo_total = sol3.t[-1]
            masa_final = sol3.y[4, -1]
        


    end_time = time.time()
    tiempo_corrida = end_time - start_time
 
    detalles = {
        'phi': phi,
        'jacobi_threshold': jacobi_threshold,
        'd0': d0,
        'tiempo_total': tiempo_total/days,
        'masa_final': masa_final,
        'exito': exito,
        'SOI': SOI_flag,
        'time_exec': tiempo_corrida

    }
    return detalles

####################################
# Iterar 
####################################
def iterar_parametros():
    phis = np.linspace(0, 360, 1440) #[110.51,110.52,175, 90,80, 70, 50  ]#np.linspace(0, 360, 720)
    #phis = [50,110.51,110.52  ]
    jacobi_thresholds = [-1.61]#np.linspace(-1.66, -1.60, 3) ### -1.63907788 -1.63907788 -1.63907788 
    d0_vals = [37000] #np.linspace(45000, 50001, 2)
    
    # Crear todas las combinaciones 
    parametros = list(product(phis, jacobi_thresholds, d0_vals))
    
    resultados=[]
    async_results = []

    with mp.Pool(processes=mp.cpu_count()-2) as pool:
        for param in parametros:
            async_result = pool.apply_async(trayectoria, args=(param,), callback=save_result)
            async_results.append(async_result)
            #resultados = pool.map(trayectoria, parametros)
        for async_result in async_results:
            try:
                res = async_result.get()
                resultados.append(res)
            except Exception as e:
                print('Error iter ', e)
    '''resumen = []
    for res in resultados:
        resumen.append({
            'phi': res['phi'],
            'jacobi_threshold': res['jacobi_threshold'],
            'd0': res['d0'],
            'tiempo_total': res['tiempo_total'],
            'masa_final': res['masa_final'],
            'exito': res['exito'],
            'SOI': res['SOI'],
            'time_exec': res['time_exec']
        })'''
    df = pd.DataFrame(resultados)
    df.to_csv('vresultados_trayectoriav003b.csv', index=False)
    print(df)
    return df

if __name__ == '__main__':
    df_resultados = iterar_parametros()
    print(df_resultados)
