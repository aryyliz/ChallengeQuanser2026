#------------------------------------------------------------------------------#
#---------------------------Vision-based Steering------------------------------#
#------------------------------------------------------------------------------#
# Implements a steering controller for a differential-drive robot
# to follow a line detected from a downward facing camera

#===============================================================================
#region: Python imports
#-------------------------------------------------------------------------------
import cv2
import os
import sys
import subprocess
import numpy as np
from pal.utilities.probe import Probe
from pal.utilities.timing import Timer
from pal.products.sensors import SensorsCamera, SensorsDisplay, SensorsTrainer
from pal.products.actuators import ActuatorsTrainer

# imports from parent directory
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from mobile_robot_functions import MDLVision, MDLDisplayFormat, \
    MDLMovement, MDLObstacle
# endregion

#===============================================================================
#region: Initialize constants and variables
#-------------------------------------------------------------------------------
TOTAL_RUNTIME = 3000 
FRAME_RATE = 30
MAIN_LOOP_RATE = 90
IMAGE_SIZE = [640,480]
ENC_RATIO = 2*np.pi/(4096*27) 

sampleCount, frameCount = 0, 0 
centroid_loc = np.array([0,0], dtype=int)
x, y, w, h = 0,0,0,0
steerError = 0
leftWheelPos, rightWheelPos = 0, 0
leftWheelSpd, rightWheelSpd = 0, 0
fwdSpdCmd, turnSpdCmd= 0, 0
leftSpdCmd, rightSpdCmd = 0, 0
obstacleDetected = False
#endregion

#===============================================================================
#region: Scope setup 
#-------------------------------------------------------------------------------
probe = Probe(ip = 'localhost')
probe.add_scope(numSignals = 2, name = 'Motor Position')
probe.add_scope(numSignals = 2, name = 'Motor Velocity')
probe.add_scope(numSignals = 2, name = 'Angles' )

subprocess.Popen(
    [sys.executable, 
     os.path.join(os.path.dirname(__file__), 'observer_steering.py')],
    cwd=os.path.dirname(__file__))
while not probe.connected:
    probe.check_connection()
# endregion

#===============================================================================
#region: Initialization
#-------------------------------------------------------------------------------
timer = Timer(sampleRate=MAIN_LOOP_RATE, totalTime=TOTAL_RUNTIME)
vision = MDLVision(IMAGE_SIZE[0], IMAGE_SIZE[1])
display = MDLDisplayFormat(IMAGE_SIZE[0], IMAGE_SIZE[1])
movement = MDLMovement(whRadius = 0.068, whDist = 0.12)
obstacle = MDLObstacle()

line2SpdMap_mdl = movement.error_to_turn_cmd(sampleTime=1/FRAME_RATE, 
                                             saturation = 2*np.pi)
next(line2SpdMap_mdl)

with (SensorsCamera(frameWidth = IMAGE_SIZE[0],
                    frameHeight = IMAGE_SIZE[1],
                    frameRate = FRAME_RATE,
                    cameraID = 0) as camera, 
        ActuatorsTrainer(block = 3) as leftMotor,
        ActuatorsTrainer(block = 1) as rightMotor, 
        SensorsDisplay() as lcd,
        SensorsTrainer() as sensors):

    leftMotor.enable_motors()
    rightMotor.enable_motors()
#endregion

    while timer.check():

        currentTime = timer.get_current_time()

        #=======================================================================
        #region: Update motor commands 
        #-----------------------------------------------------------------------
        (leftSpdCmd, rightSpdCmd) = movement.inverse_velocity_kinematics(fwdSpdCmd,  
                                                                         turnSpdCmd)

        if obstacleDetected:
            leftSpdCmd = 0
            rightSpdCmd = 0
        
        leftMotor.update_dc(-leftSpdCmd, limitCmd=False) 
        rightMotor.update_dc(rightSpdCmd, limitCmd=False) 
        leftMotor.write_motors()
        rightMotor.write_motors()

        leftMotor.read_outputs()
        rightMotor.read_outputs()
        leftWheelPos  = ENC_RATIO*leftMotor.encoder
        rightWheelPos = ENC_RATIO*rightMotor.encoder 
        leftWheelSpd = ENC_RATIO*leftMotor.tach
        rightWheelSpd = ENC_RATIO*rightMotor.tach
        #endregion

        #=======================================================================
        #region: Update scopes & Obstacles
        #-----------------------------------------------------------------------
        if sampleCount%4==0:
            probe.send(name='Motor Position', scopeData=(currentTime, [leftWheelPos, rightWheelPos]))
            probe.send(name='Motor Velocity', scopeData=(currentTime, [leftWheelSpd, rightWheelSpd]))
            probe.send(name='Angles', scopeData = (currentTime, [steerError, turnSpdCmd]))

        sensors.read_outputs()
        TOFMeasurements = (sensors.TOFDistance * sensors.TOFNumberOfTargets).reshape(8,8) 
        TOFCenterDist = np.mean((TOFMeasurements[3:4,3:4]).flatten())
        
        obstacleDetected = obstacle.detect_obstacle(dist = TOFCenterDist, 
                                                    speed = fwdSpdCmd, 
                                                    threshold = 0.05)
        #endregion

        #=======================================================================
        #region: Image preprocessing and line detection
        #-----------------------------------------------------------------------
        newFrame = camera.read()
        if newFrame:
            image = camera.imageData

            imageThreshold = vision.hsv_threshold(image,
                                                    hsvLow=np.array([15,65,150]), 
                                                    hsvHigh=np.array([40,220,255]))
            
            # EL TRUCO DEL OFFSET: Mirar solo la franja inferior (ras de piso)
            y_start = 350
            y_end = IMAGE_SIZE[1] # 480

            # Bajamos el minArea a 300 porque el recorte de la imagen ahora es mucho más pequeño
            col, row, left, top, width, height = vision.image_find_objects(
                                                    imageThreshold[y_start:y_end, :], 
                                                    connectivity = 8, 
                                                    minArea = 300, 
                                                    maxArea = 30000)
        # endregion

            #===================================================================
            #region: Calculate steering command
            #-------------------------------------------------------------------
            
            # CORRECCIÓN VITAL: Quitamos el "and frameCount%5 == 0". 
            # ¡El cálculo del error DEBE hacerse a 30 FPS, sin pausas, para que el PID no se vuelva loco!
            if col is not None:
                centroid_loc = [int(col), int(row) + y_start]
                x, y, w, h = left, top + y_start, width, height
                steerError = vision.centroid_to_steering_error(col, row)

            # 1. Control PID (Corriendo a 30Hz reales. Kp=0.5 es más seguro, Kd=0.12 amortigua)
            turnSpdCmd, _, _, _ = line2SpdMap_mdl.send((steerError, 0.5, 0.0, 0.12))
            
            # 2. Control Dinámico de Velocidad
            MAX_SPEED = 0.04   # Velocidad en recta (Si el robot es estable, puedes subir a 0.05)
            MIN_SPEED = 0.015  # Velocidad mínima al girar
            
            # Factor de frenado. Al detectar curva, bajará la velocidad suavemente.
            fwdSpdCmd = MAX_SPEED - (abs(steerError) * 0.00015)
            fwdSpdCmd = max(MIN_SPEED, min(MAX_SPEED, fwdSpdCmd))
            #endregion

            #===================================================================
            #region: Update displays
            #-------------------------------------------------------------------
            # Dejamos la restricción de 5 frames SOLO para los gráficos, para no saturar el procesador
            if frameCount%5 ==0:
                
                if col is not None:
                    image = display.draw_bounding_box(image,centroid_loc, x,y,w,h)
                image = display.draw_arrow(image,turnSpdCmd)
            
                lcd.draw_image(image)
                lcd.print_text(f"Steer Error: {steerError:.1f}", 40, 2, setColor=True, rgb = [255,0,0])
                lcd.print_text(f"Turn Spd: {turnSpdCmd:.1f}", 40, 6, setColor=True, rgb = [0,0,255])
                lcd.print_text(f"Fwd Spd: {fwdSpdCmd:.3f}", 40, 10, setColor=True, rgb = [0,255,0])

                cv2.waitKey(1)
            #endregion
            frameCount += 1

        sampleCount += 1
        timer.sleep()

cv2.destroyAllWindows()
probe.terminate()

input ("Press Enter to exit...")
