from flask import Response
from flask import Flask
from flask import render_template
import threading
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from imports.utils.datasets import LoadStreams
from imports.utils.general import check_img_size, non_max_suppression, scale_coords
from pathlib import Path
from imports.centroidtracker import CentroidTracker
import Drone_Commands

DLTA_DIF = 30
W = 320
H = 240
SOURCE = "0"
WEIGHTS = "yolov5s.pt"
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

drone = Drone_Commands.cmd()
ct = CentroidTracker()

outputFrame = None
lock = threading.Lock()


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
                    #drone.goto(0, 0.1)
            elif W / 2 > centroid[0]:
                if abs(centroid[0] - W / 2) < DLTA_DIF:
                    print("W left in the center")
                else:
                    print("go left")
                    #drone.goto(0, -0.1)
            else:
                print("x in the center")

            if H / 2 < centroid[1]:
                if abs(centroid[0] - H / 2) < DLTA_DIF:
                    print("W right in the center")
                else:
                    print("go up")
                    #drone.goto_position_target_local_ned(0, 0, -0.1)
            elif H / 2 > centroid[1]:
                if abs(centroid[0] - H / 2) < DLTA_DIF:
                    print("W right in the center")
                else:
                    print("go down")
                    #drone.goto_position_target_local_ned(0, 0, 0.1)
            else:
                print("y in the center")
            # detectionArea = abs(endX-startX) * abs(endY-startY)
            # if detectionArea > 30:
            #     print("go back")
            # elif detectionArea < 60:
            #     print("go forward")
            # else:
            #     print("At the right distance")

def capture_v():
    global outputFrame, lock
    device = torch.device("cpu")
    model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
    imgsz = check_img_size(IMGSZ, s=model.stride.max())  # check img_size

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(SOURCE, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names

    fps = 0
    start_time = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=[0],
                                   agnostic=False)  # classes = [0] mean that i want only the person detection from the weights
        rects = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x0, y0 = int(xyxy[0].item()), int(xyxy[1].item())
                    x1, y1 = int(xyxy[2].item()), int(xyxy[3].item())
                    box = (x0, y0, x1, y1)
                    (startX, startY, endX, endY) = box
                    cv2.rectangle(im0, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    rects.append(box)

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if objectID == 0:
                    for i in range(len(rects)):
                        (startX, startY, endX, endY) = rects[i]
                        if int((startX + endX) / 2.0) == centroid[0] and int((startY + endY) / 2.0) == centroid[1]:
                            break
                    cv2.rectangle(im0, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                else:
                    cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            tracking(objects, rects)

        with lock:
            outputFrame = im0 #copy the image with the results to stream to web
        fps += 1
        if fps == 200:
            end_time = time.time()
            print("cam FPS:", fps / (end_time - start_time))
            start_time = time.time()
            fps = 0



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
    #drone.connect()
    #drone.arm_and_takeoff(3)
    #drone.set_groundspeed(5)
    try:
        # start a thread that will perform motion detection
        t = threading.Thread(target=capture_v, args=())
        t.daemon = True
        t.start()

        # start the flask app
        app.run(host="0.0.0.0", port=8000, debug=True,
                threaded=True, use_reloader=False)
    except:
        pass
        #drone.land()


if __name__ == '__main__':
    main()
