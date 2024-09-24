import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Enable GPU growth to avoid allocation issues
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load the YOLO model and weights (for both Video and Webcam options)
print("Loading YOLO model...")
net = cv2.dnn.readNet('yolov3-custom_7000.weights', "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print("YOLO model loaded!")

# Load the helmet detection model (for both Video and Webcam options)
print("Loading helmet detection model...")
model = load_model('helmet-nonhelmet_cnn.h5')
print('Helmet detection model loaded!')

# Colors for helmet detection results
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for helmets, Red for no helmets

# Helper function to predict helmet or no-helmet
def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        prediction = model.predict(helmet_roi)[0][0]
        return int(prediction > 0.3)  # Adjust threshold as needed
    except Exception as e:
        print(f"Error in helmet classification: {e}")
        return 0  # Default to "helmet" if there's an error

# Function for processing video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Processing video...")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter('output.avi', fourcc, 5, (888, 500))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        ret, img = cap.read()
        if not ret:
            print("Finished processing the video or could not read frame.")
            break

        img = imutils.resize(img, height=500)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        classIds = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if classIds[i] == 0:
                    helmet_roi = img[max(0, y):y + h, max(0, x):x + w]
                    c = helmet_or_nohelmet(helmet_roi)
                    color = COLORS[c]
                    label = 'no-helmet' if c == 1 else 'helmet'
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        writer.write(img)
        cv2.imshow("Helmet Detection", img)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

# Function for processing webcam input
def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        ret, img = cap.read()
        if not ret:
            print("Finished processing webcam input.")
            break

        img = imutils.resize(img, height=500)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        classIds = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))

                    x = min(x, width - 20)
                    y = min(y, height - 20)
                    w = min(w, width - x)
                    h = min(h, height - y)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                color = [int(c) for c in COLORS[classIds[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                helmet_roi = img[y:y + h // 2, x:x + w]
                prob = helmet_or_nohelmet(helmet_roi)

                if prob < 0.5:
                    label = 'Helmet'
                    label_color = COLORS[0]
                else:
                    label = 'No-Helmet'
                    label_color = COLORS[1]

                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), label_color, 2)

        cv2.imshow("Helmet Detection", img)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI setup with Tkinter
def start_interface():
    root = tk.Tk()
    root.title("Helmet Detection")
    root.geometry("400x300")
    root.configure(bg="#2c3e50")

    def choose_video():
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi")])
        if video_path:
            process_video(video_path)

    def choose_webcam():
        process_webcam()

    def exit_app():
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            root.destroy()

    label = tk.Label(root, text="Helmet Detection System", bg="#2c3e50", fg="white", font=("Arial", 24, "bold"))
    label.pack(pady=20)

    video_button = tk.Button(root, text="Select Video", command=choose_video, font=("Arial", 15), width=20, bg="#27ae60", fg="white")
    video_button.pack(pady=10)

    webcam_button = tk.Button(root, text="Use Webcam", command=choose_webcam, font=("Arial", 15), width=20, bg="#2980b9", fg="white")
    webcam_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", command=exit_app, font=("Arial", 15), width=20, bg="#e74c3c", fg="white")
    exit_button.pack(pady=10)


    root.mainloop()

# Start the interface
if __name__ == "__main__":
    start_interface()
