import cv2
import json
from datetime import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, Entry, Button
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading

class CarDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Single-Camera Car Detection App")

        # Camera URL Entry
        self.camera_url_entry = Entry(root, width=50)
        self.camera_url_entry.grid(row=0, column=0, padx=10, pady=5)

        # Start Button
        start_button = Button(root, text="Start", command=self.start_detection)
        start_button.grid(row=0, column=1, padx=10, pady=5)

        # Model setup
        self.model = YOLO("yolov8n.pt")

        # Capture setup 
        self.cap = None
        self.thread = None
        self.running = False

        # Variables for GUI updatess
        self.car_count = tk.StringVar()
        self.previous_car_count = 0

        # Label to display car count
        self.label = ttk.Label(root, textvariable=self.car_count)
        self.label.grid(row=1, column=0, padx=10, pady=5)

        # Display Frame
        self.frame_label = ttk.Label(root)
        self.frame_label.grid(row=2, column=0, padx=10, pady=5)

        # Delay for GUI update in milliseconds
        self.gui_update_delay = 100

    def start_detection(self):
        camera_url = self.camera_url_entry.get()

        if camera_url:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(camera_url)
            self.running = True

            # Start a new thread for video capture and processing
            self.thread = threading.Thread(target=self.update)
            self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                results = self.model.predict(frame)
                car_info_list = []

                for box in results[0].boxes:
                    class_id = results[0].names[box.cls[0].item()]
                    if class_id == 'car':
                        x, y, w, h = box.xyxy[0].tolist()
                        confidence = round(box.conf[0].item(), 2)

                        car_info = {
                            "object_type": class_id,
                            "confidence": confidence,
                            "bounding_box": {
                                "x": int(x),
                                "y": int(y),
                                "width": int(w),
                                "height": int(h)
                            }
                        }

                        car_info_list.append(car_info)

                car_count_value = len(car_info_list)
                self.car_count.set(f"Car Count: {car_count_value}")

                if car_count_value != self.previous_car_count:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    count_info = {
                        "date_time": current_time,
                        "car_count": car_count_value,
                        "cars": car_info_list
                    }

                    with open("car_count.json", "a") as json_file:
                        json.dump(count_info, json_file)
                        json_file.write("\n")

                    self.previous_car_count = car_count_value

                # Schedule the display_frame method after a delay
                self.root.after(self.gui_update_delay, lambda: self.display_frame(frame))

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)

        car_count_text = f"{self.car_count.get()}"
        date_time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overlay_text = f"{car_count_text}\n{date_time_text}"

        font = ImageFont.load_default()
        text_color = (255, 255, 255)

        text_position = (10, 10)

        draw.text(text_position, overlay_text, font=font, fill=text_color)

        image = ImageTk.PhotoImage(image)

        self.frame_label.imgtk = image
        self.frame_label.configure(image=image)

    def stop_detection(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

root = tk.Tk()
app = CarDetectionApp(root)
root.mainloop()

# Ensure that the video capture is released when the application exits
app.stop_detection()
if app.cap is not None:
    app.cap.release()