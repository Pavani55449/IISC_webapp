import cv2

thres = 0.45  # Threshold to detect object

# Load the image
image_path = r"C:\Users\premk\Downloads\89447050.jpg"  # Replace 'your_image.jpg' with the path to your image
img = cv2.imread(image_path)

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']  # Vehicle classes from COCO

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=thres)

vehicle_counts = {vehicle: 0 for vehicle in vehicle_classes}  # Initialize counts for each type of vehicle

if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        className = classNames[classId - 1]
        if className in vehicle_classes:
            vehicle_counts[className] += 1  # Increment count for detected vehicle type

            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the counts for each type of vehicle
for vehicle, count in vehicle_counts.items():
    print("Number of", vehicle, "detected:", count)
