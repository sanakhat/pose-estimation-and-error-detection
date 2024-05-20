import cv2 as cvv

try:
    weights = cvv.dnn.readNetFromTensorflow("graph_opt.pb")
except cvv.error as e:
    print("Error loading model:", e)

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

# Read the image
try:
    img = cvv.imread(r"C:\Users\Admin\OneDrive\Desktop\WhatsApp Image 2024-05-11 at 13.18.03_550d4d5f.jpg")
except cvv.error as e:
    print("Error loading image:", e)

# Function for human pose estimation
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

# Call the pose estimation function with the loaded image
human_pose_estimation(img)

# Display the image with the pose estimation
cvv.imshow("Pose Estimation", img)
cvv.waitKey(0)
cvv.destroyAllWindows()
