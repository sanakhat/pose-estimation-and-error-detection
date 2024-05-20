import cv2 as cvv
import numpy as np
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the model
try:
    weights = cvv.dnn.readNetFromTensorflow("graph_opt.pb")
except cvv.error as e:
    print("Error loading model:", e)

Width = 368
Height = 368
th = 0.2

# Define body parts and connections
PARTS = {"Nose": 0, "Neck": 1, "RightShoulder": 2, "RightElbow": 3, "RightWrist": 4,
         "LeftShoulder": 5, "LeftElbow": 6, "LeftWrist": 7, "RightHip": 8, "RightKnee": 9,
         "RightAnkle": 10, "LeftHip": 11, "LeftKnee": 12, "LeftAnkle": 13, "RightEye": 14,
         "LeftEye": 15, "RightEar": 16, "LeftEar": 17, "Background": 18}

PAIRS = [["Neck", "RightShoulder"], ["Neck", "LeftShoulder"], ["RightShoulder", "RightElbow"],
         ["RightElbow", "RightWrist"], ["LeftShoulder", "LeftElbow"], ["LeftElbow", "LeftWrist"],
         ["Neck", "RightHip"], ["RightHip", "RightKnee"], ["RightKnee", "RightAnkle"], ["Neck", "LeftHip"],
         ["LeftHip", "LeftKnee"], ["LeftKnee", "LeftAnkle"], ["Neck", "Nose"], ["Nose", "RightEye"],
         ["RightEye", "RightEar"], ["Nose", "LeftEye"], ["LeftEye", "LeftEar"]]

# Function for human pose estimation
def human_pose_estimation(image):
    IWidth = image.shape[1]
    IHeight = image.shape[0]
    weights.setInput(cvv.dnn.blobFromImage(image, 1.0, (Width, Height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    o = weights.forward()
    o = o[:, :19, :, :]
    assert len(PARTS) == o.shape[1]

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
        assert partF in PARTS
        assert partT in PARTS

        idF = PARTS[partF]
        idT = PARTS[partT]

        if pnts[idF] and pnts[idT]:
            cvv.line(image, pnts[idF], pnts[idT], (0, 255, 0), 3)
            cvv.ellipse(image, pnts[idF], (3, 3), 0, 0, 360, (0, 0, 255), cvv.FILLED)
            cvv.ellipse(image, pnts[idT], (3, 3), 0, 0, 360, (0, 0, 255), cvv.FILLED)

    t, _ = weights.getPerfProfile()
    frequency = cvv.getTickFrequency() / 1000
    cvv.putText(image, '%.2fms' % (t / frequency), (10, 20), cvv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return image, pnts

# Calculate Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Calculate error rate between two sets of key points
def calculate_error_rate(points1, points2, image_data, webcam_data):
    total_error = 0
    valid_pairs = 0
    error_rates = []
    for pair in PAIRS:
        partF = pair[0]
        partT = pair[1]
        idF = PARTS[partF]
        idT = PARTS[partT]
        if points1[idF] and points1[idT] and points2[idF] and points2[idT]:
            error_rate = (euclidean_distance(points1[idF], points2[idF]) + euclidean_distance(points1[idT], points2[idT])) / 2
            total_error += error_rate
            valid_pairs += 2
            error_rates.append(error_rate)

    if valid_pairs == 0:
        return 0
    average_error_rate = total_error / 100*valid_pairs
    
    with open('error_rates.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([total_error, valid_pairs, average_error_rate])
        
    return average_error_rate

# Create Tkinter root window
root = Tk()
root.withdraw()  # Hide the root window

# Ask the user to select an image file
image_file_path = askopenfilename()

# Open the selected image file
if image_file_path:
    img = cvv.imread(image_file_path)
else:
    raise FileNotFoundError("No image file selected or path invalid.")

# Open CSV file and write headers
with open('error_rates.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Total Error', 'Valid Pairs', 'Error Rate'])

# Perform pose estimation on the selected image
static_image, static_points = human_pose_estimation(img.copy())

# Display the static image with pose estimation
cvv.imshow("Static Pose Estimation", static_image)

# Open webcam
capture = cvv.VideoCapture(0)
if not capture.isOpened():
    raise IOError("Cannot Open Webcam")

# Set up window for webcam pose estimation
window_name = 'Pose Estimation Using Webcam'

while True:
    hasimage, webcam_image = capture.read()
    if not hasimage:
        break

    # Resize webcam image to 50% of the screen
    webcam_image = cvv.resize(webcam_image, (int(webcam_image.shape[1] * 1.2), int(webcam_image.shape[0] * 1.5)))
    
    # Perform pose estimation on the webcam image
    webcam_image, webcam_points = human_pose_estimation(webcam_image)

    # Calculate error rate between the poses
    error_rate = calculate_error_rate(static_points, webcam_points, 'Static Image', 'Webcam')

    # Display webcam image with pose estimation and error rate
    cvv.putText(webcam_image, f'Error Rate: {error_rate:.2f}', (10, 40), cvv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Display webcam image with pose estimation
    cvv.imshow(window_name, webcam_image)

    # Check for ESC key or window close event
    key = cvv.waitKey(1)
    if key == 27:  # ESC key
        break

# Release resources and close windows
capture.release()
cvv.destroyAllWindows()
