import cv2

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

while camera.isOpened():
    ret, img = camera.read()

    if ret:
        # 수평으로 뒤집기
        img_flipped = cv2.flip(img, 0)

        cv2.imshow('myimg', img_flipped)

        # 'c' 키를 누르면 현재 프레임을 이미지로 저장하고 루프를 종료합니다.
        if cv2.waitKey(1) == ord('c'):
            cv2.imwrite('captured_image.jpg', img_flipped)
            break

        # 'q' 키를 누르면 루프를 종료합니다.
        elif cv2.waitKey(1) == ord('q'):
            break
    else:
        print("Error capturing frame")
        break

camera.release()
cv2.destroyAllWindows()
