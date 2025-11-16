import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "cam.mp4"
cap = cv2.VideoCapture(video_path)

def detect_circle_and_color(frame, bbox):
    """
    Detect circles (ไฟ traffic light) เฉพาะ ROI ของ bounding box
    และตรวจสอบสี (แดง/เขียว)
    """
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # Hough Circle สำหรับไฟเล็ก
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=20,
        minRadius=2,
        maxRadius=60
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cx, cy, r = c
            center = (x1 + cx, y1 + cy)
            # วาด circle
            cv2.circle(frame, center, r, (0, 255, 0), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), 3)

            # ตรวจสอบสีภายในวง
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

            # แปลงเป็น HSV
            hsv = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2HSV)
            # เฉลี่ยค่า hue ของ pixel ใน mask
            h, s, v = cv2.split(hsv)
            h_mean = cv2.mean(h, mask=mask)[0]

            label = "UNKNOWN"

            # กำหนด label ตาม hue (คร่าว ๆ)
            if 0 <= h_mean <= 10 or 160 <= h_mean <= 180:
                label = "RED"
                print("RED light detected")
            elif 40 <= h_mean <= 85:
                label = "GREEN"
                print("GREEN light detected")

            cv2.putText(frame, label, (center[0]-10, center[1]-r-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Loop through video frames
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
                detect_circle_and_color(annotated_frame, (x1, y1, x2, y2))

    cv2.imshow("YOLO + Traffic Light Color", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
