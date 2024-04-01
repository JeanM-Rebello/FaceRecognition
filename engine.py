import os, os.path
import face_recognition as fr

class Recognition():
    def __init__(self):
        self.recognized_faces = []
        self.name_faces = []
        self.filling()
    
    def face_recognition(self,photho_folder_path,photo):
        self.photo = fr.load_image_file(f"{photho_folder_path}/{photo}")
        self.faces = fr.face_encodings(self.photo)
        if(len(self.faces)>0):
            return True,self.faces
        return False, []

    def get_face_picture(self,photo_folder_path,photo_folder):
        for images in range(len(photo_folder)):
            sucess,face = self.face_recognition(photo_folder_path,photo_folder[images])
            if(sucess):
                self.recognized_faces.extend(face)
                self.name_faces.append(os.path.basename(os.path.normpath(photo_folder_path)))
        return self.recognized_faces, self.name_faces

    def filling(self):
        chadwick_path = "./img/Chadwick"
        chadwick_photos = os.listdir(chadwick_path)
        chris_path = "./img/Chris"
        chris_photos = os.listdir(chris_path)
        megan_path = "./img/Megan"
        megan_photos = os.listdir(megan_path)

        self.get_face_picture(megan_path,megan_photos)
        self.get_face_picture(chadwick_path,chadwick_photos)
        self.get_face_picture(chris_path,chris_photos)