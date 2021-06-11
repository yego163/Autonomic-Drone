import face_recognition
import cv2
import numpy as np
from multiprocessing import Process, Queue
import time
import queue
import winsound

#1356236087

IMAGE_SHRINK_RATIO = 1 #how much to shrink the image frame, the more we shrink the more FPS we can process BUT the face must be closer to the camera!
Q_SIZE = 10
PROCESS_NUM = 5 # with hog use 5, with cnn use max 3 otherwise we run out of GPU mem
SKIP_FRAMES = 2


class FaceRec:

    pleaseStop = False
    origFrameQ=None
    finalFrameQ=None
    moveCmdQ=None

        
    def stop(self):
        self.pleaseStop=True
    
    def calcWantedPos(self,when,frame,top, right, bottom, left):
        x,y,z = 0,0,0
        (h, w) = frame.shape[:2]
        x = (w/2) - ((right-left)/2) # calc where the center should be
        z = ((right-left)) # calc box width around the face found
        x = x - left # how far are we from the center
        if x > -110 and x < 80: # if we are close to the center, don't move
            x = 0
        y = (h/2) - ((bottom-top)/2) # cal where the center should be
        y = y - top # how far are we from the center
        if y > -70 and y < 60: # if we are close to the center, don't move
           y = 0

        if z>80 and z<110:
            z=90 # make sure we don't move the drone back/forward
            
        try:
            if ((x!=0 or y != 0 or z !=90) and self.moveCmdQ.empty()):
                self.moveCmdQ.put_nowait((when,x,y,z))
                print (x,y,z)
        except queue.Full:
            pass
            #print("**** CMD Q is full ****")            
        
        
        
        
    def consumeOrigFrame(self,q):
        start = time.time()
        frameCount=0
        fps=0
        process_this_frame = 0
        while(self.pleaseStop != True):
            when, frame = q.get()
           
            # from bgr to rgb 
            rgb_small_frame = frame[:, :, ::-1]

            # Only process every SKIP_FRAMES frame of video to save time
            
            if process_this_frame>=SKIP_FRAMES:
                frameCount = frameCount + 1
                # Find all the faces and face encodings in the current frame of video
                
                self.face_locations = face_recognition.face_locations(rgb_small_frame, model="hog") # model="cnn" or model="hog" (faster but less accurate)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
              
                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    self.face_names.append(name)
                process_this_frame = 0
            else:    
                process_this_frame = process_this_frame + 1
                
            
            
            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):

                if name=="Wanted":
                    # put the wanted face rec location in cmd Q
                    self.calcWantedPos(when,frame,top, right, bottom, left)
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    
                    # Draw a label with a name below the face
                    #cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left, bottom + 25), font, 1.0, (0, 0, 255), 1)
          
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            self.finalFrameQ.put((when,frame))
            
            now = time.time()
            if now-start>=5:              
                fps=frameCount/(now-start) 
                #print("Proceesed FPS: {:.2f}".format(fps))
                frameCount=0
                start = time.time()

    def consumeFinalFrame(self,q):
        start = time.time()
        frameCount=0
        fps=0
        lastFrameTime=0
        while(self.pleaseStop != True):
            #name = threading.currentThread().getName()
            #print(name,finalFrameQ.qsize())
            when, frame = q.get();
            if lastFrameTime>when:
                continue
            lastFrameTime = when
            # do processing here...
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (2, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
            cv2.imshow('DJI Tello', frame)
            
            frameCount = frameCount + 1
            now = time.time()
            if now-start>=5:              
                fps=frameCount/(now-start) 
                #print("Final FPS: {:.2f}".format(fps))
                frameCount=0
                start = time.time()
            # Video Stream is closed if escape key is pressed
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            #q.task_done()
            
        cv2.destroyAllWindows()

    def __init__(self, frameQ,cmdQ):
        
        # Load a sample picture and learn how to recognize it.
        wanted_image = face_recognition.load_image_file("wanted.jpg")
        wanted_face_encoding = face_recognition.face_encodings(wanted_image)[0]

        # Load a second sample picture and learn how to recognize it.
        biden_image = face_recognition.load_image_file("uk.jpg")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            wanted_face_encoding,
            biden_face_encoding
        ]
        self.known_face_names = [
            "Wanted",
            "Unknown"
        ]

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # start al  l consumer producer threads
        self.origFrameQ = frameQ
        self.moveCmdQ = cmdQ
        self.finalFrameQ = Queue(maxsize = Q_SIZE)
        for i in range(PROCESS_NUM):
            p = Process(name = "FaceRecProcess"+str(i),target=self.consumeOrigFrame, args=(self.origFrameQ,))
            p.start()

        p=Process(name = "finalThread", target=self.consumeFinalFrame, args=(self.finalFrameQ,))
        p.start()

