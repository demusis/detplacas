import cv2
import numpy as np
import time

cars_detect = cv2.CascadeClassifier('haarcascade_car.xml')

capture = cv2.VideoCapture('cars.avi') # <----

while capture.isOpened():

    time.sleep(.05)

    # Comeca a capturar os frames
    ret, frame = capture.read()
    
    # Cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # cv2.CascadeClassifier.detectMultiScale(imagem, scaleFactor, minNeighbors)
    # Verificar parametro Minsize
    carros_detectados = cars_detect.detectMultiScale(cinza, 1.1, 1)
    
    # cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    for (x,y,w,h) in carros_detectados:
        cor_roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)      
        cv2.imshow('Carros detectados', frame)
        
        data_hora = time.strftime("%Y%m%d-%H%M%S")
        cd_1 = cv2.imwrite('cd-' + data_hora + '.jpg', cor_roi)

    c = cv2.waitKey(1)
    if c == 27:
       break

capture.release()

cv2.destroyAllWindows()
