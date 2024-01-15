import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import RPi.GPIO as GPIO
import threading
import atexit
import time

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

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def cleanup():
    stop_all_motors()
    GPIO.cleanup()
    camera.release()
    cv2.destroyAllWindows()



vertices = np.array([[(0, 480), (100, 300), (500, 300), (640, 480)]], dtype=np.int32)

# class RoadClassifier(nn.Module):
#     def __init__(self, num_classes=3):
#         super(RoadClassifier, self).__init__()
#         self.cnn = models.resnet18(pretrained=True)
#         in_features = self.cnn.fc.in_features
#         self.cnn.fc = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         return self.cnn(x)
    
# 간단한 CNN 모델 정의
class RoadClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(RoadClassifier, self).__init__()
        #self.cnn = models.resnet18(pretrained=True)
        self.cnn = models.mobilenet_v3_small(pretrained=True)
        # in_features = self.cnn.fc.in_features #ResNet18
        # self.cnn.fc = nn.Linear(in_features, num_classes)

        in_features = self.cnn.classifier[-1].in_features #mobilenet_v2
        self.cnn.classifier[-1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.cnn(x)

# 모델 초기화
model = RoadClassifier(num_classes=3).to(torch.device('cpu'))

# 저장된 가중치 불러오기
checkpoint = torch.load('road_classifier_model (12).pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

camera = cv2.VideoCapture(0)
image = None  # 캡처된 이미지를 저장할 변수
capture_lock = threading.Lock()

def capture_frames():
    global image
    while True:
        _, captured_frame = camera.read()
        captured_frame = cv2.flip(captured_frame, -1)
        # 프레임 처리 작업
        masked_frame = region_of_interest(captured_frame, vertices)

        # 락을 이용하여 이미지에 동시에 접근하지 않도록 함
        with capture_lock:
            #image = masked_frame.copy()
            image = cv2.resize(masked_frame, (224, 224))

    return captured_frame

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

while True:
    # 이미지 전처리
    with capture_lock:
        if image is None:
            continue  # 이미지가 캡처되지 않은 경우 스킵

        input_frame = image.copy()
    # 모델 예측 시간 측정 시작
    start_time = time.time()

    # transform 및 모델 예측
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(input_frame)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        result = model(input_batch)
        predict = torch.argmax(result).item()
        print('result:', result)
        print("class:", predict)

    # 모델 예측 시간 측정 종료
    end_time = time.time()
    prediction_time = end_time - start_time

    print('Prediction time:', prediction_time, 'seconds')

    # 화면에 결과 표시
    cv2.putText(image, f'Prediction: {predict}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Road Classifier', image)

    # 모터 제어
    if predict == 0:
        print("Go")
        set_motor_speed('A', 40)
        set_motor_speed('B', 50)
    elif predict == 1:
        print("Left")
        set_motor_speed('A', 40)
        set_motor_speed('B', 25)
    elif predict == 2:
        print('Right')
        set_motor_speed('A', 15)
        set_motor_speed('B', 50)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

atexit.register(cleanup())