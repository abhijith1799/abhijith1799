import cv2
import numpy as np
import face_recognition


#part- conversion
imageAbhi = face_recognition.load_image_file('img/train.jpg')
imageAbhi = cv2.cvtColor(imageAbhi,cv2.COLOR_BGR2RGB)
imagetest = face_recognition.load_image_file('img/elon.jpeg')
imagetest = cv2.cvtColor(imagetest,cv2.COLOR_BGR2RGB)




# Part 2 locating and encoding
facelocationAbhi = face_recognition.face_locations(imageAbhi)[0]
faceencodeAbhi = face_recognition.face_encodings(imageAbhi)[0]
cv2.rectangle(imageAbhi,(facelocationAbhi[3],facelocationAbhi[0]),(facelocationAbhi[1],facelocationAbhi[2]),(255,0,255),2,0,0)

facelocationtest = face_recognition.face_locations(imagetest)[0]
faceencodetest = face_recognition.face_encodings(imagetest)[0]
cv2.rectangle(imagetest,(facelocationtest[3],facelocationtest[0]),(facelocationtest[1],facelocationtest[2]),(255,0,255),2,0,0)

results = face_recognition.compare_faces([faceencodeAbhi],faceencodetest)

facedistace = face_recognition.face_distance([faceencodeAbhi],faceencodetest)
print(results,facedistace)
cv2.putText(imagetest,f'{results}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('Abhijith',imageAbhi)
cv2.imshow('Abhijith test',imagetest)
cv2.waitKey(0)









