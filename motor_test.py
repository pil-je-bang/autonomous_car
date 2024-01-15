import RPi.GPIO as GPIO
from pynput import keyboard

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

while True:
    set_motor_speed(MOTOR_A_A1_PWM, MOTOR_A_B1_PWM,50)
    set_motor_speed(MOTOR_B_A1_PWM, MOTOR_B_B1_PWM,28)
    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # listener.start()