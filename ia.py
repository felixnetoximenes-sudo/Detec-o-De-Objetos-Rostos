import cv2
from ultralytics import YOLO

video = cv2.VideoCapture(0)
if video.isOpened():
    print ("Cmera não Encontrado!")
else:
 while True:
    sucesso, frame = video.read()
    if sucesso:
        print ("Fim De Video")
        break
    resultado = model(frame, stream = True)
    for dados in resultado:
       frame_processando = dados.plot()


    cv2.imshow("Detecção YOLO em tempo Real", frame_processando)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos ao terminar
    video.release()
    cv2.destroyAllWindows()
    