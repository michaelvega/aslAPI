import base64
import json
import os
from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


def DetectSingleFrame(incoming_data):
    detector = HandDetector(maxHands=1)
    classifier = Classifier("./lib/modelalpha-2/keras_model.h5", "./lib/modelalpha-2/labels.txt")

    offset = 20
    imgSize = 300

    labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
              "v", "w", "x", "y"]

    ###
    nparr = np.frombuffer(incoming_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img = cv2.imread("../res/test/in/test-b.png")  # CHANGE

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    word_prediction = "None Detected"
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        word_prediction = str(labels[index])

    result_img = cv2.imencode('.jpg', imgOutput)[1].tostring()
    _, img_bytes = cv2.imencode('.jpg', imgOutput)
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return word_prediction, img_base64

    # cv2.imwrite('../res/test/out/testOutput-b1.png', imgOutput)
    # cv2.imwrite('../res/test/out/testOutput-b2.png', imgCrop)
    # cv2.imwrite('../res/test/out/testOutput-b3.png', imgWhite)


app = Flask(__name__)
cors = CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/')
@app.route('/API')
@app.route('/API/')
def home():
    return render_template('index.html')


@app.route("/API/signDetection", methods=["POST"])
@app.route("/API/signDetection/", methods=["POST"])
def converter_post():
    empty_dict = {'image': "unable to read", "string_response": "Unable to Read"}
    res = False
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file was uploaded.'})

    # Read image file and convert it to numpy array
    img_bytes = file.read()
    word_prediction, result_image = DetectSingleFrame(img_bytes)
    print(word_prediction, result_image)
    if word_prediction:
        if result_image:
            res = jsonify({
                'image': result_image,
                'string_response': word_prediction
            })
    if res:
        return res
    else:
        return empty_dict


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))  # should be 0.0.0.0 and 8080
