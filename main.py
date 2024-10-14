import cv2
import numpy as np
from time import sleep

# Parameters
largura_min = 80  
altura_min = 80   
offset = 6        
pos_linha = 550   
delay = 60        

detec_cam1 = []
detec_cam2 = []
carros_cam1 = 0
carros_cam2 = 0

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap1 = cv2.VideoCapture('video.mp4')
cap2 = cv2.VideoCapture('video.mp4')

subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

def process_frame(frame, detec, carros, pos_linha):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, pos_linha), (1200, pos_linha), (176, 130, 39), 2)

    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = get_center(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)

        for (x, y) in detec:
            if (y < (pos_linha + offset)) and (y > (pos_linha - offset)):
                carros += 1
                cv2.line(frame, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
                detec.remove((x, y))
                print(f"Car detected. Current count: {carros}")

    return frame, carros

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break

    frame1, carros_cam1 = process_frame(frame1, detec_cam1, carros_cam1, pos_linha)
    frame2, carros_cam2 = process_frame(frame2, detec_cam2, carros_cam2, pos_linha)

    if carros_cam1 > carros_cam2 + 10:
        traffic_status = "Possible Traffic Jam or Blockage"
    elif carros_cam2 > carros_cam1:
        traffic_status = "Traffic Flowing Smoothly"
    else:
        traffic_status = "Normal Traffic Flow"

    cv2.putText(frame1, f"CAM1 COUNT: {carros_cam1}", (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
    cv2.putText(frame2, f"CAM2 COUNT: {carros_cam2}", (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
    cv2.putText(frame1, f"TRAFFIC STATUS: {traffic_status}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)

    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap1.release()
cap2.release()
