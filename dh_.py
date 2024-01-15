import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import RPi.GPIO as GPIO
import threading
import paramiko


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        print("[",x,',',y,']')

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
def birdview(img):
    img = img

    pts1 = np.float32([[421,254],[229,439],[480,255],[757,429]])
    pts2 = np.float32([[0,0],[0,600],[600,0],[600,600]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(700,700))

    return dst

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
        print(ignore_mask_color)
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices,ignore_mask_color)


    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5, fill_color = [0,0,255]):
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-5)  # 기울기 계산, 분모에 작은 값 추가하여 나누기 오류 방지
            if abs(slope) > 0.5:  # 기울기가 일정 값 이하인 경우만 그리도록 설정
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    pts = np.array([[(x1, y1), (x2, y2), (x2, img.shape[0]), (x1, img.shape[0])]], dtype=np.int32)
    cv2.fillPoly(img, pts, fill_color)

def hough_lines(img1,img2, rho, theta, threshold, min_line_len, max_line_gap,origin):
    lines_left =     cv2.HoughLinesP(img1, rho, theta, threshold,np.array([]),
                                minLineLength = min_line_len,
                                maxLineGap=max_line_gap)
    lines_right =    cv2.HoughLinesP(img2, rho, theta, threshold,np.array([]),
                                minLineLength = min_line_len,
                                maxLineGap=max_line_gap)
    original = origin

    if((lines_left is None) or (lines_right is None)):
        print()
    else:
        draw_lines(original, lines_left)
        draw_lines(original,lines_right)


    return original

def send_image(image):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.35', username='pi', password='0000')

    # 이미지를 바이트 배열로 변환
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    # SSH를 통해 이미지 전송
    sftp = ssh.open_sftp()
    with sftp.file('remote_image.jpg', 'wb') as f:
        f.write(img_bytes)

    sftp.close()
    ssh.close()

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

vertices = np.array([[(0,480),(100, 300), (500, 300), (640,480)]], dtype=np.int32)

class RoadClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(RoadClassifier, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)

# 모델 초기화
model = RoadClassifier(num_classes=3).to(torch.device('cpu'))

# 저장된 가중치 불러오기
checkpoint = torch.load('road_classifier_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

def capture_frames():
    while True:
        ret, frame = cap.read()
        # 프레임 처리 작업

cap = cv2.VideoCapture(0)
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# 카메라 설정
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

while camera.isOpened():
    _, image = camera.read()
    image = cv2.flip(image, -1)
    send_image(image)

    mask=region_of_interest(image,vertices)


    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(mask)
    input_batch = input_tensor.unsqueeze(0)

    # 모델 예측
    with torch.no_grad():
        result = model(input_batch)
        predict = torch.argmax(result).item()
        print('result:', result)
        print("class:", predict)

    # 화면에 결과 표시
    cv2.putText(image, f'Prediction: {predict}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Road Classifier', image)

    # 모터 제어
    if predict == 0:
        print("Go")
        set_motor_speed('A', 100)
        set_motor_speed('B', 100)
    elif predict == 1:
        print("Left")
        set_motor_speed('A', 100)
        set_motor_speed('B', 50)
    elif predict == 2:
        print('Right')
        set_motor_speed('A', 50)
        set_motor_speed('B', 100)



    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
# 프로그램 종료 시 정리
stop_all_motors()
GPIO.cleanup()