import torch
import cv2
import numpy as np
import RPi.GPIO as GPIO

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


import torch.nn as nn
from torchvision import models

class RoadClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(RoadClassifier, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)

# 모델 초기화
model = RoadClassifier(num_classes=3)

# 저장된 가중치 불러오기
checkpoint = torch.load('road_classifier_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# 카메라 설정
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

while camera.isOpened():
    _, image = camera.read()
    image = cv2.flip(image, 0)

    # 이미지 전처리
    height, width, _ = image.shape
    save_image = image[int(height/3*2):, :, :]
    save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
    save_image = cv2.GaussianBlur(save_image, (3, 3), 0)
    save_image = cv2.resize(save_image, (200,66))
    x = np.asarray([save_image])
    x = torch.tensor(x).to(torch.device('cpu'))
    
    # 모델 예측
    with torch.no_grad():
        result=model(x)
        print("result",result)
        predict = torch.argmax(result).item()

    # 모터 제어
    if predict == 0:
        print("Go")
        set_motor_speed('A', 20)
        set_motor_speed('B', 20)
    elif predict == 1:
        print("Left")
        set_motor_speed('A', 20)
        set_motor_speed('B', 0)
    elif predict == 2:
        print('Right')
        set_motor_speed('A', 0)
        set_motor_speed('B', 20)

# 프로그램 종료 시 정리
stop_all_motors()
GPIO.cleanup()
