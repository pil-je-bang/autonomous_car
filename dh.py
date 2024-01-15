# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import cv2
import math
# %matplotlib inline

#grayscale
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#blur
def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
#canny
def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)
#관심영역 설정
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

def draw_lines(img,lines,color=[255,0,0],thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)

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

while camera.isOpened():
    ret, img = camera.read()
    img = cv2.flip(img, 0)
    
    gray=grayscale(img)
    blur_gray=gaussian_blur(gray,kernel_size)
    edges=canny(blur_gray,low_threshold,high_threshold)
    mask=np.zeros_like(img)

    if len(img.shape)>2:
        channel_count=img.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color=255

    imshape=img.shape
    vertices=np.array([[(0,imshape[0]),(0,imshape[0]/2),(imshape[1],imshape[0]/2),(imshape[1],imshape[0])]],dtype=np.int32)
    mask=region_of_interest(edges,vertices)

    print(imshape)
    lines=hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)
    lines_edges=weighted_img(lines,img,a=0.8,b=1.,c=0.)

    cv2.imshow('myimg', lines_edges)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()