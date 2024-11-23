import tkinter as tk
from tkinter import Label, Button
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
model = load_model('/home/doaa/programming/ML/ML/Project/ModelTesting/model_file_30epochs.h5')
import sys
# sys.stdout.reconfigure(encoding='utf-8')

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    faceDetect=cv2.CascadeClassifier('/home/doaa/programming/ML/ML/Project/ModelTesting/haarcascade_frontalface_default.xml')

    labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    if not cap.isOpened():
        print("Unable to open the camera")
    else:
        print("Camera open successfully")
    while cap.isOpened():
        ret, frame = cap.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= faceDetect.detectMultiScale(gray, 1.3, 3)
        for x,y,w,h in faces:
            sub_face_img=gray[y:y+h, x:x+w]
            resized=cv2.resize(sub_face_img,(48,48))
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 48, 48, 1))
            result=model.predict(reshaped)
            label=np.argmax(result, axis=1)[0]
            print(label)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        # if ret :
        cv2.imshow("Camera", frame)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def close_camera():
    global cap
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    
root = tk.Tk()
root.title("COMSOC Camera Application")
root.geometry("1000x600")

# photo = tk.PhotoImage(file="/home/doaa/programming/ML/ML/Project/ieee.png")
# label = tk.Label(root,image=photo)
# label.place(x=0, y=0, relwidth=1, relheight=1)



img = Image.open("/home/doaa/programming/ML/ML/Project/ieee.png")

# Convert the image to something Tkinter can use
img_tk = ImageTk.PhotoImage(img)

label = tk.Label(root, image=img_tk)
label.place(x=0, y=0, relwidth=1, relheight=1)


# label.image = photo



root.configure(bg= "black")
button_frame = tk.Frame(root,bg="black")
button_frame.pack(side="bottom", pady=20, padx=50,anchor="center") 


btn_open = Button(root,text="Open Camera",
                  command=lambda:threading.Thread(target=open_camera).start(),
                  bg="green", fg="white", font=("Helvetica", 14, "bold"), padx=10, pady=10)
btn_open.pack(in_=button_frame, side="left", padx=20)


btn_close = Button(root,
                   text="Close Camera",
                   command=close_camera,
                   bg="red", fg="white", font=("Helvetica", 14, "bold"), padx=10, pady=10)
btn_close.pack(in_=button_frame, side="left", padx=10)


# btn_exit = tk.Button(root, text="Exit", command=root.quit, bg="black", fg="white", font=("Helvetica", 14, "bold"), padx=10, pady=10)
# btn_exit.pack(in_=button_frame, side="left", padx=10)

lbl_video = tk.Label(root)
lbl_video.pack(pady=20)

root.mainloop()