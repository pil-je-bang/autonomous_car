import cv2
import RPi.GPIO as GPIO
import time
from pynput import keyboard
import threading

# 카메라 및 녹화 설정
def camera_setup():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    # 동영상 녹화 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 또는 다른 코덱

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 동영상 크기 변경

    while camera.isOpened():
        ret, img = camera.read()

        if ret:
            img_flipped = cv2.flip(img, 1)
            cv2.imshow('myimg', img_flipped)
            out.write(img_flipped)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Error capturing frame")
            break

    camera.release()
    out.release()
    cv2.destroyAllWindows()

# 모터 제어 설정
def motor_setup():
    # 모터 핀 설정
    MOTOR_PINS = {
        'A': {'A1': 5, 'B1': 6},
        'B': {'A1': 20, 'B1': 21}
    }

    GPIO.setmode(GPIO.BCM)
    for motor in MOTOR_PINS.values():
        for pin in motor.values():
            GPIO.setup(pin, GPIO.OUT)

    MOTOR_PWM = {}
    for motor_label, motor in MOTOR_PINS.items():
        MOTOR_PWM[motor_label] = {
            'A1': GPIO.PWM(motor['A1'], 1000),
            'B1': GPIO.PWM(motor['B1'], 1000)
        }
        MOTOR_PWM[motor_label]['A1'].start(0)
        MOTOR_PWM[motor_label]['B1'].start(0)

    def set_motor_speed(motor_label, speed):
        pwm_a = MOTOR_PWM[motor_label]['A1']
        pwm_b = MOTOR_PWM[motor_label]['B1']
        if speed >= 0:
            pwm_a.ChangeDutyCycle(speed)
            pwm_b.ChangeDutyCycle(0)
        else:
            pwm_a.ChangeDutyCycle(0)
            pwm_b.ChangeDutyCycle(-speed)

    def stop_all_motors():
        for motor in MOTOR_PWM.values():
            motor['A1'].stop()
            motor['B1'].stop()

    def on_press(key):
        try:
            if key.char == 'w':
                set_motor_speed('A', 100)
                set_motor_speed('B', 100)
            elif key.char == 's':
                set_motor_speed('A', -100)
                set_motor_speed('B', -100)
            elif key.char == 'a':
                set_motor_speed('A', 50)
                set_motor_speed('B', 100)
            elif key.char == 'd':
                set_motor_speed('A', 100)
                set_motor_speed('B', 50)
        except AttributeError:
            pass

    def on_release(key):
        if key == keyboard.Key.esc:
            return False
        else:
            set_motor_speed('A', 0)
            set_motor_speed('B', 0)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_all_motors()
        GPIO.cleanup()

# 멀티스레딩으로 카메라 및 모터 제어 동시 실행
camera_thread = threading.Thread(target=camera_setup)
motor_thread = threading.Thread(target=motor_setup)

camera_thread.start()
motor_thread.start()

camera_thread.join()
motor_thread.join()
