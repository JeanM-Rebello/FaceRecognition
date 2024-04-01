import face_recognition as fr 
from engine import Recognition

import os

recog = Recognition()

unknown_path = "./img/Unknown"
unknown_photos = os.listdir(unknown_path)

sucess,unknown = recog.face_recognition(unknown_path,unknown_photos[0])
if(sucess):
    unknown_face = unknown[0]
    known_face, name_face = recog.recognized_faces, recog.name_faces
    results = fr.compare_faces(known_face, unknown_face)
    print(results)
    for faces in range(len(known_face)):
        result = results[faces]
        if(result):
            print("Rosto do", name_face[faces], "foi reconhecido")