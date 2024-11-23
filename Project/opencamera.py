import tkinter as tk 
from tkinter import Label, ttk
import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
model = load_model('/home/doaa/programming/ML/ML/Project/ModelTesting/model_file_30epochs.h5')

camera_running = False
cap = None

def open_camera():
    global camera_running, cap
    if not camera_running:
        cap = cv2.VideoCapture(0)  # to open the camera 
        faceDetect=cv2.CascadeClassifier('/home/doaa/programming/ML/ML/Project/ModelTesting/haarcascade_frontalface_default.xml')
        labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

        camera_running = True
        show_frame()

def stop_camera():
    global camera_running, cap
    if camera_running and cap:
        camera_running = False
        cap.release() 
        lbl_video.config(image='') # delete any displayed image 

def show_frame():
    global camera_running, cap
    if camera_running:
        ret, frame = cap.read()
        if ret:
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.config(image=imgtk)
        lbl_video.after(10, show_frame) 


root = tk.Tk()
root.title("COMSOC Camera Application")
root.geometry("1000x600")


bg_image = tk.PhotoImage(file="/home/doaa/programming/ML/ML/Project/ieee.png")  
lbl_bg = tk.Label(root, image=bg_image)
lbl_bg.place(x=0, y=0, relwidth=1, relheight=1)


root.configure(bg= "black")
button_frame = tk.Frame(root,bg="black")
button_frame.pack(side="bottom", pady=20, padx=50,anchor="center")  


btn_open = tk.Button(root, text="Open the Camera", command=open_camera , bg="green", fg="white", font=("Helvetica", 14, "bold"), padx=10, pady=10)
btn_open.pack(in_=button_frame, side="left", padx=20)

btn_stop = tk.Button(root, text="Close the Camera", command=stop_camera, bg="red", fg="white", font=("Helvetica", 14, "bold"), padx=10, pady=10)
btn_stop.pack(in_=button_frame, side="left", padx=10)


btn_exit = tk.Button(root, text="Exit", command=root.quit, bg="black", fg="white", font=("Helvetica", 14, "bold"), padx=10, pady=10)
btn_exit.pack(in_=button_frame, side="left", padx=10)



lbl_video = tk.Label(root)
lbl_video.pack(pady=20)


root.mainloop()
