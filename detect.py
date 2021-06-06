from flask import Response
from flask import Flask
from flask import render_template
import threading
import time

import cv2
import os

from imports.centroidtracker import CentroidTracker
from imports.common import input_size
from imports.detect import get_objects
from imports.edgetpu import make_interpreter
from imports.edgetpu import run_inference
import Drone_Commands

DLTA_DIF = 30
W = 320
H = 240
drone = Drone_Commands.cmd()

default_model_dir = '/home/pi/dev/new_human_detect_with_fly_drone'
default_model = 'mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite'  # works faster for person detection
default_labels = 'coco_labels.txt'

model = os.path.join(default_model_dir, default_model)
label = os.path.join(default_model_dir, default_labels)
top_k = 3  # number of categories with highest score to display
camera_idx = 0  # Index of which video source to use.
threshold = 0.6  # classifier score threshold

outputFrame = None
lock = threading.Lock()
ct = CentroidTracker()

# initialize a flask object
app = Flask(__name__)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def tracking(objects, rects):
    first_object = True
    for (objectID, centroid) in objects.items():
        if first_object:
            first_object = False
            for i in range(len(rects)):
                (startX, startY, endX, endY) = rects[i]
                if int((startX + endX) / 2.0) == centroid[0] and int((startY + endY) / 2.0) == centroid[1]:
                    break
            if W / 2 < centroid[0]:
                if abs(centroid[0] - W / 2) < DLTA_DIF:
                    print("W right in the center")
                else:
                    print("go right")
                    drone.goto(0, 0.1)
            elif W / 2 > centroid[0]:
                if abs(centroid[0] - W / 2) < DLTA_DIF:
                    print("W left in the center")
                else:
                    print("go left")
                    drone.goto(0, -0.1)
            else:
                print("x in the center")

            if H / 2 < centroid[1]:
                if abs(centroid[0] - H / 2) < DLTA_DIF:
                    print("W right in the center")
                else:
                    print("go up")
                    drone.goto_position_target_local_ned(0, 0, -0.1)
            elif H / 2 > centroid[1]:
                if abs(centroid[0] - H / 2) < DLTA_DIF:
                    print("W right in the center")
                else:
                    print("go down")
                    drone.goto_position_target_local_ned(0, 0, 0.1)
            else:
                print("y in the center")
            # detectionArea = abs(endX-startX) * abs(endY-startY)
            # if detectionArea > 30:
            #     print("go back")
            # elif detectionArea < 60:
            #     print("go forward")
            # else:
            #     print("At the right distance")



def append_objs_to_img(cv2_im, inference_size, objs):
    global objects
    rects = []
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.id == 0:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
            box = (x0, y0, x1, y1)
            rects.append(box)
            (startX, startY, endX, endY) = box
            cv2.rectangle(cv2_im, (startX, startY), (endX, endY), (0, 255, 0), 2)
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(cv2_im, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if objectID == 0:
            for i in range(len(rects)):
                (startX, startY, endX, endY) = rects[i]
                if int((startX + endX) / 2.0) == centroid[0] and int((startY + endY) / 2.0) == centroid[1]:
                    break
            cv2.rectangle(cv2_im, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.circle(cv2_im, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
        else:
            cv2.circle(cv2_im, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    tracking(objects, rects)
    return cv2_im


def capture_v():
    global outputFrame, lock

    print('Loading {} with {} labels.'.format(model, label))
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(camera_idx)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # Sony PS3 EYE cam settings:
    # 320x240 @ 125 FPS, 640x480 @ 60 FPS, 320x240 @187 FPS --> use excat FSP setting
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320),
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("image size=", size)

    fps = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2_im = cv2.flip(frame, -1)
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, threshold)[:top_k]
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs)
        with lock:
            outputFrame = cv2_im
        fps += 1
        if fps == 200:
            end_time = time.time()
            print("cam FPS:", fps / (end_time - start_time))
            start_time = time.time()
            fps = 0
    cap.release()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    fps = 0
    start_time = time.time()
    # loop over frames from the output stream
    while True:
        fps += 1
        if fps == 200:
            end_time = time.time()
            print("web FPS:", fps / (end_time - start_time))
            start_time = time.time()
            fps = 0
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    drone.connect()
    drone.arm_and_takeoff(3)
    drone.set_groundspeed(5)
    try:
        # start a thread that will perform motion detection
        t = threading.Thread(target=capture_v, args=())
        t.daemon = True
        t.start()

        # start the flask app
        app.run(host="0.0.0.0", port=8000, debug=True,
                threaded=True, use_reloader=False)
    except:
        drone.land()


if __name__ == '__main__':
    main()
