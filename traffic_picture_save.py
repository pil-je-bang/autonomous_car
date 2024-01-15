import numpy as np
import cv2
import RPi.GPIO as GPIO
import time
from pynput import keyboard
#import math
# %matplotlib inline

MOTOR_A_A1=5
MOTOR_A_B1=6
MOTOR_B_A1=20
MOTOR_B_B1=21

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_A_A1,GPIO.OUT)
GPIO.setup(MOTOR_A_B1,GPIO.OUT)
GPIO.setup(MOTOR_B_A1,GPIO.OUT)
GPIO.setup(MOTOR_B_B1,GPIO.OUT)

MOTOR_A_A1_PWM=GPIO.PWM(MOTOR_A_A1,1000)
MOTOR_A_B1_PWM=GPIO.PWM(MOTOR_A_B1,1000)
MOTOR_B_A1_PWM=GPIO.PWM(MOTOR_B_A1,1000)
MOTOR_B_B1_PWM=GPIO.PWM(MOTOR_B_B1,1000)

MOTOR_A_A1_PWM.start(0)
MOTOR_A_B1_PWM.start(0)
MOTOR_B_A1_PWM.start(0)
MOTOR_B_B1_PWM.start(0)



def set_motor_speed(pwm_a, pwm_b, speed):
    if speed >= 0:
        pwm_a.ChangeDutyCycle(speed)
        pwm_b.ChangeDutyCycle(0)
    else:
        pwm_a.ChangeDutyCycle(0)
        pwm_b.ChangeDutyCycle(-speed)

def on_press(key):
    if key.char == 'w':
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM,100)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM,100)
        print('input is w')
    elif key.char == 's':
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM,-100)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM,-100)
        print('input is s')
    elif key.char == 'a':
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM, 100)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM, 50)
        print('input is a')
    elif key.char == 'd':
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM, 50)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM, 100)
        print('input is d') 

def on_release(key):
    # if key.char == 'q':
    if key == keyboard.Key.esc:
        MOTOR_A_A1_PWM.stop()
        MOTOR_A_B1_PWM.stop()
        MOTOR_B_A1_PWM.stop()
        MOTOR_B_B1_PWM.stop()
        GPIO.cleanup()
        return False
    else:
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM, 0)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM, 0)

camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

i=389
carstate = "stop"
file_path = "/home/pi/traffic/green/"
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while camera.isOpened():


    _, image = camera.read()
    image = cv2.flip(image, -1)

    cv2.imshow('main', image)
    
    height, _, _ = image.shape
    save_image = image[int(height/3*2): , :, :]
    save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
    save_image = cv2.GaussianBlur(save_image, (3,3),0)
    save_image = cv2.resize(save_image, (200,66))

    key = cv2.waitKey(1)
    if key == ord('q'):
        listener.join()
        break
    elif key == 119:
        print("go")
        carstate = "go"
        cv2.imwrite(f"{file_path}{i:05d}.png", image)
        i += 1
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM,100)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM,100)
    elif key == 115:
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM,-100)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM,-100)
        print('input is s')
    elif key == 97:
        print("left")
        carstate = "left"
        cv2.imwrite(f"{file_path}{i:05d}.png", image)
        i += 1
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM, 100)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM, 50)
        print('input is a')
    elif key == 100:
        print("right")
        carstate = "right"
        cv2.imwrite(f"{file_path}{i:05d}.png", image)
        i += 1
        set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM, 50)
        set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM, 100)
        print('input is d') 

cv2.distroyAllwindows()
