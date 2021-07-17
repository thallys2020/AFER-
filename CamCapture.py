import cv2
from functions import calculoAU, Anger, Hapiness, Surprise, Sadness, Fear, Disgust
import face_recognition


cap = cv2.VideoCapture(0)

while(True):
    #lê cada quadro do vídeo e carrega na varíavel frame
    _, frame = cap.read()
  
    cv2.imshow("Pressione S para Capturar a emocao Neutra",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'): #save on pressing 'y' 
        referencia = frame
        cv2.destroyAllWindows()
        break

cap.release()  

cap = cv2.VideoCapture(0)

while(True):
    #lê cada quadro do vídeo e carrega na varíavel frame
    _, frame = cap.read()
    
    #vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if face_recognition.face_landmarks(frame):
        
        result = calculoAU(referencia, frame)
        
        Anger(result)
        
        Hapiness(result)
        
        Surprise(result)
        
        Sadness(result)
        
        Fear(result)
        
        Disgust(result)
        
    cv2.imshow("Captura",frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 

cv2.destroyAllWindows()  


    
    
        
  
        
        
        