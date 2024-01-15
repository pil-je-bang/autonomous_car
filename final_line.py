# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
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

#grayscale
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#blur
def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
#canny
def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)
#roi
def region_of_interest(img,vertices):
    mask=np.zeros_like(img)

    if len(img.shape)>2:
        channel_count=img.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color=255

    cv2.fillPoly(mask,vertices,ignore_mask_color)
    
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-5)  # 기울기 계산, 분모에 작은 값 추가하여 나누기 오류 방지
            if abs(slope) > 0.5:  # 기울기가 일정 값 이하인 경우만 그리도록 설정
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:  # Check if lines is None
        print("No lines found!")
        return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img,initial_img,a=0.8,b=1.,c=0.):
    return cv2.addWeighted(initial_img,a,img,b,c)

kernel_size=5
low_threshold=50
high_threshold=200

rho=5
theta=np.pi/180
threshold=90
min_line_len=120
max_line_gap=150

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    ret, img = camera.read()
    img = cv2.flip(img, 0)

    gray = grayscale(img)
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    mask = np.zeros_like(img)

    imshape = img.shape
    vertices = np.array([[(0, imshape[0]), (100, imshape[0]/2), (500, imshape[0]/2), (imshape[1], imshape[0])]], dtype=np.int32)
    mask = region_of_interest(edges, vertices)

    lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)
    lines_edges = weighted_img(lines, img, a=0.8, b=1., c=0.)

    cv2.imshow('myimg', lines_edges)

    if cv2.waitKey(1) == ord('q'):
        listener.join()
        break

camera.release()
cv2.destroyAllWindows()