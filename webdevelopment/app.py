from flask import Flask, render_template, request, jsonify, url_for
import cv2
import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the Flask app


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET','POST'])
def detect():
    image_file = request.files['image']
    image_path = 'static/' + image_file.filename
    image_file.save(image_path)

    # Perform object detection
    thres = 0.45  # Threshold to detect object
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

    classNames = []
    with open('coco.names', 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = r'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

    weightsPath = r'frozen_inference_graph.pb'

    img = cv2.imread(image_path)
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    vehicle_counts = {vehicle: 0 for vehicle in vehicle_classes}
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in vehicle_classes:
                vehicle_counts[className] += 1

    # Draw bounding boxes on the image
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    # Save the detection image
    detection_image_path = 'static/detection_' + image_file.filename
    cv2.imwrite(detection_image_path, img)

    # Prepare data for response
    car_count = vehicle_counts['car']
    truck_count = vehicle_counts['truck']
    bus_count = vehicle_counts['bus']
    motorcycle_count = vehicle_counts['motorcycle']
    bicycle_count = vehicle_counts['bicycle']
    response_data = {
        'car_count': car_count,
        'truck_count':truck_count,
        'bus_count': bus_count,
        'motorcycle_count': motorcycle_count,
        'bicycle_count': bicycle_count,
     
        'original_image_path': url_for('static', filename=image_file.filename),
        'detection_image_path': url_for('static', filename='detection_' + image_file.filename)
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)