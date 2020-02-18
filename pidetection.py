from flask import Flask, request, render_template, Response
import datetime as Hari
from datetime import datetime
import webbrowser
import numpy as np
import face_recognition
import cv2
from pyimagesearch.motion_detection import SingleMotionDetector
import threading
from threading import Timer
import argparse
import imutils
import pickle

from imutils.video import VideoStream
from imutils.video import FPS

app = Flask(__name__)


outputFrame = None
lock = threading.Lock()

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

def detect_motion(frameCount):
    frame_number = 0
    global cap, outputFrame, lock, data

    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    print("[INFO] loading encodings")
    detector = cv2.CascadeClassifier(args["cascade"])

    fps = FPS().start()

    while True:
        data = pickle.loads(open(args["encodings"], "rb").read())
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # face detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # face recognition

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []


        for (top, right, bottom, left), face_encoding in zip(boxes, encodings):

            matches = face_recognition.compare_faces(data["encodings"],
            face_encoding, tolerance=0.4)

            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = data["names"][best_match_index]
                status = True

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            # cv2.rectangle(img, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                face_image = frame[top:bottom, left:right]
                cv2.imwrite( "person_found/" + name + '_Face.jpg', face_image)          

            else:

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        timestamp = Hari.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        
        if total > frameCount:
            motion = md.detect(gray)
            if motion is not None:
                (thresh, (minX, minY, maxX, maxY)) = motion
                # cv2.rectangle(img, (minX, minY), (maxX, maxY),(0, 0, 255), 2)
        
        md.update(gray)
        total += 1

        with lock:
            outputFrame = frame.copy()


def generate():
    
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/')
def index(): 
        return render_template('recognition.html')

@app.route('/resetAll')
def resetAll():               
        return render_template('recognition.html')

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

# run Flask app
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--frame-count", type=int, default=32,)
    ap.add_argument("-c", "--cascade", default='haarcascade_frontalface_default.xml')
    ap.add_argument("-e", "--encodings", default='encodings.pickle')
    args = vars(ap.parse_args())
    args = vars(ap.parse_args())

    
    t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
    t.daemon = True
    t.start()
    #app.run(debug=True, threaded=True, use_reloader=False, host='192.168.0.132', port=5000)
    Timer(1, open_browser).start();
    app.run(debug=True, threaded=True, use_reloader=False)

# vs.stop()