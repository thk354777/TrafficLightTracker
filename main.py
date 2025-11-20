import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
video_path = "cam.mp4"
cap = cv2.VideoCapture(video_path)

# Temporal smoothing
frame_labels = deque(maxlen=100)

def detect_traffic_light_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Red mask
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Green mask
    lower_green = np.array([40,50,50])
    upper_green = np.array([85,255,255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    label = "UNKNOWN"

    def check_mask(mask, color_name):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 500:
                ((cx, cy), r) = cv2.minEnclosingCircle(cnt)
                mask_roi = cv2.bitwise_and(roi, roi, mask=mask)
                _, s, v = cv2.split(cv2.cvtColor(mask_roi, cv2.COLOR_BGR2HSV))
                s_mean = cv2.mean(s, mask=mask)[0]
                v_mean = cv2.mean(v, mask=mask)[0]
                if s_mean > 50 and v_mean > 50:
                    # Background text
                    text = color_name
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
                    x_text = int(cx)+x1
                    y_text = int(cy)+y1 - int(r) - 5
                    cv2.rectangle(frame,
                                (x_text, y_text - text_height - baseline),
                                (x_text + text_width, y_text + baseline),
                                (0,0,0), cv2.FILLED)
                    cv2.putText(frame, text, (x_text, y_text), font, scale, (255,255,255), thickness)
                    return color_name
        return None

    res = check_mask(mask_red, "RED")
    if res is not None:
        label = res
    else:
        res = check_mask(mask_green, "GREEN")
        if res is not None:
            label = res

    frame_labels.append(label)
    most_common = max(set(frame_labels), key=frame_labels.count)
    return most_common

def detect_lanes(frame):
    """
    Lane detection using Canny + ROI + Hough Transform
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI: ครึ่งล่างของ frame
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(0, int(height/1.3)), (width, int(height/1.3)), (width, height), (0, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)
    lane_frame = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_frame, (x1,y1), (x2,y2), (0,255,0), 2)
    return lane_frame

# Loop video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    for result in results:
        boxes = result.boxes
        scores = result.boxes.conf
        for box, cls, score in zip(boxes.xyxy, boxes.cls, scores):
            x1, y1, x2, y2 = map(int, box)
            if int(cls) == 9 and score > 0.3:
                detect_traffic_light_color(annotated_frame, (x1, y1, x2, y2))

    # Lane detection window
    lane_frame = detect_lanes(frame)

    cv2.imshow("Traffic Light Detection", annotated_frame)
    cv2.imshow("Lane Detection", lane_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
