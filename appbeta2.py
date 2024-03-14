import cv2
import pandas as pd
from ultralytics import YOLO
import imutils
from imutils.video import VideoStream
import tkinter as tk
from tkinter import simpledialog

class Tracker:
    def __init__(self):
        pass

    def update(self, bbox_list):
        # Your tracking logic here
        pass

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Create a Tkinter window to get RTSP URL from the user
root = tk.Tk()
root.withdraw()  # Hide the main window

# Ask the user for the RTSP URL using a dialog box
rtsp_url = simpledialog.askstring("Input", "Enter RTSP URL:")
if rtsp_url is None:
    exit()

# Start the video stream with the provided RTSP URL
video_stream = VideoStream(rtsp_url).start()

# Add a check to ensure the video capture is opened successfully
if not video_stream.stream.isOpened():
    print("Error: Unable to open the RTSP stream.")
    exit()

model = YOLO('yolov8s.pt')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()

cy1 = 322
cy2 = 368
offset = 6

while True:
    frame = video_stream.read()
    if frame is None:
        continue

    count += 1
    if count % 1 != 0:  # Adjust frame skipping here (in this case, every 5th frame is processed)
        continue
    frame = imutils.resize(frame, width=1020)

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    bbox_list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            bbox_list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(bbox_list)

    # Check if bbox_id is not None before iterating
    if bbox_id is not None:
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("RGB", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
video_stream.stop()

