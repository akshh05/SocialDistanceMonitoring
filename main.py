import cv2
import numpy as np
import imutils
from twilio.rest import Client  

def calculate_distance(point1, point2):
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    print(f"Distance calculated: {distance}")  # Debug print
    return distance

def load_yolo_model(weights_path, cfg_path):
    try:
        net = cv2.dnn.readNet(weights_path, cfg_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        print("YOLO model loaded successfully.")
        return net, output_layers
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        exit()

def load_coco_names(names_path):
    try:
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print("COCO names loaded successfully.")
        return classes
    except Exception as e:
        print(f"Error loading COCO names: {e}")
        exit()

def send_sms_alert():
    account_sid = "AC2c36f245898ae17bfa3c66a7b1bb71d5"
    auth_token = "9721158231f2c38789276d4ebe325700"
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body="Alert! Social distancing violation detected.",
        from_='+18302641261',  
        to='+917904733787'  
    )

    print(f"SMS alert sent: {message.sid}")

def main():
    weights_path = "yolov3.weights"
    cfg_path = "yolov3.cfg"
    names_path = "coco.names"

    net, output_layers = load_yolo_model(weights_path, cfg_path)
    classes = load_coco_names(names_path)

    DISTANCE_THRESHOLD = 25  # Adjust this threshold according to your needs
    cap = cv2.VideoCapture(0)  # Use webcam

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        exit()

    frame_count = 0
    max_frames = 100  

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            frame_count += 1
            if frame_count > max_frames:
                print("Max frames reached. Exiting.")
                break

            frame = imutils.resize(frame, width=800)
            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            if not outputs:
                print("No outputs received from YOLO.")
                continue

            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to 'person'
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            if not boxes:
                print("No boxes detected.")
                continue

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(indexes) > 0:
                points = []
                alert_triggered = False
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    points.append((x + w // 2, y + h // 2))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        distance = calculate_distance(points[i], points[j])
                        if distance < DISTANCE_THRESHOLD:
                            cv2.line(frame, points[i], points[j], (0, 0, 255), 2)
                            cv2.putText(frame, "Alert!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            alert_triggered = True

                if alert_triggered:
                    print("Alert triggered!")
                    send_sms_alert()

            cv2.imshow("Social Distancing Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
