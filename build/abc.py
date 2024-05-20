import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cvv
import csv
import matplotlib.pyplot as plt
import subprocess
def open_webcam():
    subprocess.Popen(["python", "check.py"])

def detect_human(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained cascade classifier for human detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If faces are detected, return True, otherwise return False
    return len(faces) > 0

def select_photo():
    filename = filedialog.askopenfilename(title="Select a Photo", filetypes=[("Image Files", ".jpg .png")])
    if filename:
        selected_image = cv2.imread(filename)
        if detect_human(selected_image):
            print("The selected image contains a human.")
            try:
                weights = cvv.dnn.readNetFromTensorflow("graph_opt.pb")
                
                Width = 368
                Height = 368
                th = 0.2

                PARTS = { "Nose": 0, "Neck": 1, "RightShoulder": 2, "RightElbow": 3, "RightWrist": 4,
                            "LeftShoulder": 5, "LeftElbow": 6, "LeftWrist": 7, "RightHip": 8, "RightKnee": 9,
                            "RightAnkle": 10, "LeftHip": 11, "LeftKnee": 12, "LeftAnkle": 13, "RightEye": 14,
                            "LeftEye": 15, "RightEar": 16, "LeftEar": 17, "Background": 18 }

                PAIRS = [ ["Neck", "RightShoulder"], ["Neck", "LeftShoulder"], ["RightShoulder", "RightElbow"],
                            ["RightElbow", "RightWrist"], ["LeftShoulder", "LeftElbow"], ["LeftElbow", "LeftWrist"],
                            ["Neck", "RightHip"], ["RightHip", "RightKnee"], ["RightKnee", "RightAnkle"], ["Neck", "LeftHip"],
                            ["LeftHip", "LeftKnee"], ["LeftKnee", "LeftAnkle"], ["Neck", "Nose"], ["Nose", "RightEye"],
                            ["RightEye", "RightEar"], ["Nose", "LeftEye"], ["LeftEye", "LeftEar"] ]

                def human_pose_estimation(image):
                    IWidth = image.shape[1]
                    IHeight = image.shape[0]
                    weights.setInput(cvv.dnn.blobFromImage(image, 1.0, (Width, Height), (127.5, 127.5, 127.5), swapRB = True, crop = False))
                    o = weights.forward()
                    o = o[:, :19, :, :]
                    assert(len(PARTS) == o.shape[1])

                    pnts = []
                    for i in range(len(PARTS)):
                        Map = o[0, i, :, :]
                        _, conf, _, point = cvv.minMaxLoc(Map)
                        X = (IWidth * point[0]) / o.shape[3]
                        Y = (IHeight * point[1]) / o.shape[2]
                        pnts.append((int(X), int(Y)) if conf > th else None)

                    for pair in PAIRS:
                        partF = pair[0]
                        partT = pair[1]
                        assert(partF in PARTS)
                        assert(partT in PARTS)

                        idF = PARTS[partF]
                        idT = PARTS[partT]

                        if pnts[idF] and pnts[idT]:
                            cvv.line(image, pnts[idF], pnts[idT], (0, 255, 0), 3)
                            cvv.ellipse(image, pnts[idF], (3, 3), 0, 0, 360, (0, 0, 255), cvv.FILLED)
                            cvv.ellipse(image, pnts[idT], (3, 3), 0, 0, 360, (0, 0, 255), cvv.FILLED)

                    t, _ = weights.getPerfProfile()
                    frequency = cvv.getTickFrequency() / 1000
                    cvv.putText(image, '%.2fms' % (t / frequency), (10, 20), cvv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    return image

                human_pose_image = human_pose_estimation(selected_image)

                cvv.imshow("Pose Estimation", human_pose_image)
                cvv.waitKey(0)
                cvv.destroyAllWindows()
                
            except cvv.error as e:
                print("Error loading model:", e)
        else:
            print("The selected image does not contain a human.")
            messagebox.showinfo("No Human Detected", "The selected image does not contain a human.")
            return

def open_csv():
    subprocess.Popen(["python","graph.py"])

def go_back(window):
    window.destroy()
    root.deiconify()

def close_app():
    root.destroy()

root = tk.Tk()
root.title("Angle Detection App")
root.attributes('-fullscreen', True)

background_image = Image.open("human-body-models-pose-estimation-ezgif.com-webp-to-jpg-converter (3).jpg")
window_width = root.winfo_screenwidth()
window_height = root.winfo_screenheight()
background_image = background_image.resize((window_width, window_height))

background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

webcam_image = Image.open(r"camera.png")
webcam_image = webcam_image.resize((100, 100))
webcam_photo = ImageTk.PhotoImage(webcam_image)

photo_image = Image.open(r"gallary.png")
photo_image = photo_image.resize((100, 100))
photo_photo = ImageTk.PhotoImage(photo_image)

csv_image = Image.open(r"csv.png")
csv_image = csv_image.resize((100, 100))
csv_photo = ImageTk.PhotoImage(csv_image)

close_image = Image.open(r"power.png")
close_image = close_image.resize((100, 100))
close_photo = ImageTk.PhotoImage(close_image)

webcam_button = tk.Button(root, image=webcam_photo, command=open_webcam, bd=0, highlightthickness=0, relief="flat")
webcam_button.place(relx=0.68, rely=0.5, anchor="center")

select_photo_button = tk.Button(root, image=photo_photo, command=select_photo, bd=0, highlightthickness=0, relief="flat")
select_photo_button.place(relx=0.27, rely=0.5, anchor="center")

open_csv_button = tk.Button(root, image=csv_photo, command=open_csv, bd=0, highlightthickness=0, relief="flat")
open_csv_button.place(relx=0.46, rely=0.5, anchor="center")

close_app_button = tk.Button(root, image=close_photo, command=close_app, bd=0, highlightthickness=0, relief="flat")
close_app_button.place(relx=0.95, rely=0.07, anchor="center")

root.mainloop()
