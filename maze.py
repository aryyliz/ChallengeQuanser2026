#------------------------------------------------------------------------------#
#--------------------Maze Solver: Radar Inicial + Deflector 5°-----------------#
#------------------------------------------------------------------------------#
# 1. Al arrancar, escanea izquierda y derecha para encontrar la salida.
# 2. Al avanzar, si detecta pared, da giros de 5° hacia el lado con mas
#    espacio, manteniendo la direccion elegida hasta librar el obstaculo.

import os
import sys
import numpy as np
from pal.utilities.timing import Timer
from pal.products.sensors import SensorsTrainer
from pal.products.actuators import ActuatorsTrainer

# Importar funciones de cinematica del proyecto
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from mobile_robot_functions import MDLMovement

#===============================================================================
#region: Parametros Ajustables
#-------------------------------------------------------------------------------
TOTAL_RUNTIME  = 600         # Tiempo maximo (s)
MAIN_LOOP_RATE = 60          # Frecuencia de control (Hz)
DT = 1.0 / MAIN_LOOP_RATE
RAD_TO_DEG = 180.0 / np.pi

# Bias del giroscopio en Z
GYRO_BIAS_Z = 0.0898

# 1. UMBRALES DE DISTANCIA (m)
FRONT_STOP_DIST  = 0.22      # 22 cm: Frena con mucha anticipacion para que al rotar no raspe las esquinas
SIDE_WALL_DIST   = 0.25      # 25 cm: Rango para autocentrarse
WALL_TARGET_DIST = 0.14      # 14 cm: Distancia ideal de la pared

# 2. PARAMETROS DE MOVIMIENTO
INIT_SWEEP_ANGLE = 50.0      # Grados que girara a cada lado al inicio para escanear
DEFLECT_ANGLE    = 5.0       # Grados por cada "mordisco" al esquivar
TURN_SPEED       = 0.40      # Velocidad de giro
FWD_SPEED        = 0.015     # Velocidad de avance super baja (m/s)
KP_WALL          = 0.85      # Fuerza del autocentrado

# Estados de la Maquina de Estados (FSM)
STATE_INIT_RIGHT = 0         # Escanea hacia la derecha
STATE_INIT_LEFT  = 1         # Escanea hacia la izquierda
STATE_INIT_ALIGN = 2         # Se apunta hacia la mejor salida encontrada
STATE_FORWARD    = 3         # Avanza
STATE_EVALUATE   = 4         # Decide hacia donde esquivar
STATE_TURN_5DEG  = 5         # Da el mordisco de 5 grados
#endregion

#===============================================================================
#region: Funcion de Filtrado Anti-Reflejos
#-------------------------------------------------------------------------------
def get_zone_distance(dist_matrix, targets_matrix, fallback=1.5):
    valid_mask = (targets_matrix > 0) & (dist_matrix > 0.02) & (dist_matrix < 2.0)
    valid_dists = dist_matrix[valid_mask]
    if valid_dists.size > 0:
        return float(np.percentile(valid_dists, 25))
    return fallback
#endregion

#===============================================================================
#region: Inicializacion y Bucle Principal
#-------------------------------------------------------------------------------
timer = Timer(sampleRate=MAIN_LOOP_RATE, totalTime=TOTAL_RUNTIME)
movement = MDLMovement(whRadius=0.068, whDist=0.12)

state = STATE_INIT_RIGHT
accumulated_yaw = 0.0
best_yaw = 0.0
best_front_dist = 0.0
turn_yaw_accum = 0.0
turn_direction = 1
is_deflecting = False
sample_count = 0

with (ActuatorsTrainer(block=3) as leftMotor,
      ActuatorsTrainer(block=1) as rightMotor,
      SensorsTrainer(ultraService=1, ultraStart=0.0, ultraLength=5.0) as sensors):

    leftMotor.enable_motors()
    rightMotor.enable_motors()
    print("Iniciando Radar Inicial + Deflector de 5°. Presiona Ctrl+C para detener.")

    try:
        while timer.check():
            # 1. Lectura de sensores
            sensors.read_outputs()
            gyro_z = sensors.gyro[2] * RAD_TO_DEG - GYRO_BIAS_Z
            ultra_raw = sensors.ultraDistance
            ultra_dist = ultra_raw if (0.02 < ultra_raw < 2.5) else 1.5

            tof_dist_mat = np.flip(sensors.TOFDistance.reshape(8, 8), axis=(0, 1))
            tof_targ_mat = np.flip(sensors.TOFNumberOfTargets.reshape(8, 8), axis=(0, 1))

            left_dist  = get_zone_distance(tof_dist_mat[2:6, 0:2], tof_targ_mat[2:6, 0:2])
            tof_center = get_zone_distance(tof_dist_mat[2:6, 3:5], tof_targ_mat[2:6, 3:5])
            right_dist = get_zone_distance(tof_dist_mat[2:6, 6:8], tof_targ_mat[2:6, 6:8])

            front_dist = min(tof_center, ultra_dist)

            fwdSpdCmd = 0.0
            turnSpdCmd = 0.0

            #===================================================================
            # 2. Maquina de Estados
            #===================================================================
            
            # --- FASE 1: RADAR INICIAL ---
            if state == STATE_INIT_RIGHT:
                turnSpdCmd = -TURN_SPEED
                accumulated_yaw += gyro_z * DT
                if front_dist > best_front_dist:
                    best_front_dist = front_dist
                    best_yaw = accumulated_yaw
                
                if accumulated_yaw <= -INIT_SWEEP_ANGLE:
                    state = STATE_INIT_LEFT

            elif state == STATE_INIT_LEFT:
                turnSpdCmd = TURN_SPEED
                accumulated_yaw += gyro_z * DT
                if front_dist > best_front_dist:
                    best_front_dist = front_dist
                    best_yaw = accumulated_yaw
                
                if accumulated_yaw >= INIT_SWEEP_ANGLE:
                    state = STATE_INIT_ALIGN

            elif state == STATE_INIT_ALIGN:
                turn_diff = best_yaw - accumulated_yaw
                if abs(turn_diff) <= 3.0: 
                    state = STATE_FORWARD
                    is_deflecting = False
                else:
                    turnSpdCmd = TURN_SPEED if turn_diff > 0 else -TURN_SPEED
                    accumulated_yaw += gyro_z * DT

            # --- FASE 2: NAVEGACION Y DEFLEXION DE 5° ---
            elif state == STATE_FORWARD:
                if front_dist <= FRONT_STOP_DIST:
                    fwdSpdCmd = 0.0
                    turnSpdCmd = 0.0
                    if not is_deflecting:
                        state = STATE_EVALUATE
                    else:
                        state = STATE_TURN_5DEG
                        turn_yaw_accum = 0.0
                else:
                    fwdSpdCmd = FWD_SPEED
                    is_deflecting = False 
                    
                    if left_dist < SIDE_WALL_DIST and right_dist < SIDE_WALL_DIST:
                        wall_error = left_dist - right_dist
                    elif right_dist < SIDE_WALL_DIST:
                        wall_error = WALL_TARGET_DIST - right_dist
                    elif left_dist < SIDE_WALL_DIST:
                        wall_error = left_dist - WALL_TARGET_DIST
                    else:
                        wall_error = 0.0
                    turnSpdCmd = np.clip(KP_WALL * wall_error, -0.25, 0.25)

            elif state == STATE_EVALUATE:
                if right_dist > left_dist:
                    turn_direction = -1
                else:
                    turn_direction = 1
                
                is_deflecting = True 
                turn_yaw_accum = 0.0
                state = STATE_TURN_5DEG

            elif state == STATE_TURN_5DEG:
                turnSpdCmd = TURN_SPEED * turn_direction
                turn_yaw_accum += gyro_z * DT
                if abs(turn_yaw_accum) >= DEFLECT_ANGLE:
                    state = STATE_FORWARD

            #===================================================================
            # 3. Enviar comandos a los motores
            #===================================================================
            leftSpdCmd, rightSpdCmd = movement.inverse_velocity_kinematics(fwdSpdCmd, turnSpdCmd)
            leftMotor.update_dc(-leftSpdCmd, limitCmd=False)
            rightMotor.update_dc(rightSpdCmd, limitCmd=False)
            leftMotor.write_motors()
            rightMotor.write_motors()

            if sample_count % 15 == 0:
                nombres = ["RADAR_DER", "RADAR_IZQ", "ALINEANDO", "AVANZA", "EVALUANDO", "GIRO_5°"]
                print(f"[{nombres[state]}] Frente:{front_dist:.2f}m | Izq:{left_dist:.2f}m | Der:{right_dist:.2f}m")

            sample_count += 1
            timer.sleep()

    finally:
        leftMotor.update_dc(0, limitCmd=False)
        rightMotor.update_dc(0, limitCmd=False)
        leftMotor.write_motors()
        rightMotor.write_motors()
