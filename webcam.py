import cv2
import dlib
import numpy as np
import face_recognition as fr
from engine import Recognition

recog = Recognition()
ESQ = 27

know_faces, name_face = recog.recognized_faces, recog.name_faces

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break
    
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        results = fr.compare_faces(know_faces, face_encoding)
        
        face_distances = fr.face_distance(know_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if results[best_match_index]:
            name = name_face[best_match_index]
        else:
            name = "Unknown"
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow("Webcam Face Recognition", frame)
    
    if cv2.waitKey(1) == ESQ:
        break

webcam.release()
cv2.destroyAllWindows()
