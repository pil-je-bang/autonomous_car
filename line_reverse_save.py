import os
import cv2

# 원본 이미지가 있는 디렉토리
input_directory = "/home/pi/line"

# 결과 이미지를 저장할 디렉토리
output_directory = "/home/pi/line_reverse"

# 원하는 확장자 설정 (여기서는 '.jpg'로 설정)
file_extension = '.png'

# 디렉토리 내의 파일 목록 가져오기
file_list = [f for f in os.listdir(input_directory) if f.endswith(file_extension)]

# 반전된 이미지 저장
for file_name in file_list:
    if file_name.endswith('L' + file_extension):
        # 이미지 읽기
        img_path = os.path.join(input_directory, file_name)
        img = cv2.imread(img_path)

        # 좌우 반전
        flipped_img = cv2.flip(img, 1)

        # 결과 이미지 저장
        output_path = os.path.join(output_directory, file_name.replace('L', 'R'))
        cv2.imwrite(output_path, flipped_img)

print("좌우 반전이 완료되었습니다.")
