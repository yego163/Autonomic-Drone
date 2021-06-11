import socket
import threading
import time
import cv2
from stats import Stats
import faceRec
import imutils
from multiprocessing import Process, Queue
import queue
import winsound


Q_SIZE = 50
CMD_Q_SIZE = 1
FRAME_WIDTH = 600 # max 600 with hog 300 with cnn (and thread count 3)

def video_thread(tello_ip,Q):
    # Creating stream capture object
    cap = cv2.VideoCapture('udp://'+tello_ip+':11111')
    # Runs while 'stream_state' is True
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, FRAME_WIDTH) # for tello, 280 is the max we can have without frame drops
        try:
            #print("qsize=",origFrameQ.qsize())
            Q.put_nowait((time.time(),frame))
        except queue.Full:
            print("**** Orig Q is full ****")            
    cap.release()

class Tello:

    origFrameQ=None
    cmdQ=None
    
    
    def calculateCMD(self,frame,x,y,z):
        height, width, channels = frame.shape
        #print (height, width)
    

    def __init__(self, tello_ip: str='192.168.10.1', debug: bool=True):
        # Opening local UDP port on 8889 for Tello communication
        self.local_ip = ''
        self.local_port = 8889
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.local_ip, self.local_port))
        
        # Setting Tello ip and port info
        self.tello_ip = tello_ip
        self.tello_port = 8889
        self.tello_address = (self.tello_ip, self.tello_port)
        self.log = []

        # Intializing response thread
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        # easyTello runtime options
        self.stream_state = False
        self.MAX_TIME_OUT = 15.0
        self.debug = debug
        # Setting Tello to command mode
        self.command()

    def send_command(self, command: str, query: bool =False):
        # New log entry created for the outbound command
        self.log.append(Stats(command, len(self.log)))

        # Sending command to Tello
        self.socket.sendto(command.encode('utf-8'), self.tello_address)
        # Displaying conformation message (if 'debug' os True)
        if self.debug is True:
            print('Sending command: {}'.format(command))
            
        # Checking whether the command has timed out or not (based on value in 'MAX_TIME_OUT')
        start = time.time()
        while not self.log[-1].got_response():  # Runs while no repsonse has been received in log
            now = time.time()
            difference = now - start
            if difference > self.MAX_TIME_OUT:
                print('Connection timed out!')
                break
        # Prints out Tello response (if 'debug' is True)
        if self.debug is True and query is False:
            print('Response: {}'.format(self.log[-1].get_response()))

    def _receive_thread(self):
        while True:
            # Checking for Tello response, throws socket error
            try:
                self.response, ip = self.socket.recvfrom(1024)
                self.log[-1].add_response(self.response)
            except socket.error as exc:
                print('Socket error: {}'.format(exc))

    
    def wait(self, delay: float):
        # Displaying wait message (if 'debug' is True)
        if self.debug is True:
            print('Waiting {} seconds...'.format(delay))
        # Log entry for delay added
        self.log.append(Stats('wait', len(self.log)))
        # Delay is activated
        time.sleep(delay)
    
    def get_log(self):
        return self.log

    # Controll Commands
    def command(self):
        self.send_command('command')
    
    def takeoff(self):
        self.send_command('takeoff')

    def land(self):
        self.send_command('land')

    def processMoveCmd(self):
        lastFrameTime=0
        found = False
        yCount = 3
        while True:
            # loop until wanted is found
            while self.cmdQ.empty() and not found:
                # self.right(30)
                self.cw(10)
                time.sleep(0.1)
            # wanted found !!
            if (not found):
                winsound.PlaySound("alarm", winsound.SND_FILENAME)
            found = True    
            when,x,y,z = self.cmdQ.get()            
            #print(when,x,y,z)
            if lastFrameTime>when:
                continue
            lastFrameTime = when
            if x < 0:
                self.right(20)
            elif x > 0:
                self.left(20)
            if yCount == 3: 
                if y < 0:
                    self.down(25)
                elif y > 0:
                    self.up(25)
                yCount = 0    
            else:
                yCount+=1
            if z <= 80:
                self.forward(20)
            elif z >= 110:
                self.back(20)
                
            #time.sleep(0.3)
            
    def streamon(self):
        self.send_command('streamon')
        self.stream_state = True
        # start all processes
        self.origFrameQ = Queue(maxsize = Q_SIZE)
        self.cmdQ = Queue(maxsize = CMD_Q_SIZE)
        p=Process(name = "origProcess", target=video_thread, args=(self.tello_ip,self.origFrameQ,))
        p.start()
        # init face rec engine
        print("Init face rec engine...")
        self.fr = faceRec.FaceRec(self.origFrameQ,self.cmdQ)
        print("Done init face rec engine...")
        

    def streamoff(self):
        self.stream_state = False
        self.send_command('streamoff')

    def emergency(self):
        self.send_command('emergency')
    
    # Movement Commands
    def up(self, dist: int):
        self.send_command('up {}'.format(dist))

    def down(self, dist: int):
        self.send_command('down {}'.format(dist))

    def left(self, dist: int):
        self.send_command('left {}'.format(dist))

    def right(self, dist: int):
        self.send_command('right {}'.format(dist))
        
    def forward(self, dist: int):
        self.send_command('forward {}'.format(dist))

    def back(self, dist: int):
        self.send_command('back {}'.format(dist))

    def cw(self, degr: int):
        self.send_command('cw {}'.format(degr))
    
    def ccw(self, degr: int):
        self.send_command('ccw {}'.format(degr))

    def flip(self, direc: str):
        self.send_command('flip {}'.format(direc))

    def go(self, x: int, y: int, z: int, speed: int):
        self.send_command('go {} {} {} {}'.format(x, y, z, speed))

    def curve(self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, speed: int):
        self.send_command('curve {} {} {} {} {} {} {}'.format(x1, y1, z1, x2, y2, z2, speed))

    # Set Commands
    def set_speed(self, speed: int):
        self.send_command('speed {}'.format(speed))

    def rc_control(self, a: int, b: int, c: int, d: int):
        self.send_command('rc {} {} {} {}'.format(a, b, c, d))

    def set_wifi(self, ssid: str, passwrd: str):
        self.send_command('wifi {} {}'.format(ssid, passwrd))

    # Read Commands
    def get_speed(self):
        self.send_command('speed?', True)
        return self.log[-1].get_response()

    def get_battery(self):
        self.send_command('battery?', True)
        return self.log[-1].get_response()

    def get_time(self):
        self.send_command('time?', True)
        return self.log[-1].get_response()

    def get_height(self):
        self.send_command('height?', True)
        return self.log[-1].get_response()
    
    def get_temp(self):
        self.send_command('temp?', True)
        return self.log[-1].get_response()

    def get_attitude(self):
        self.send_command('attitude?', True)
        return self.log[-1].get_response()

    def get_baro(self):
        self.send_command('baro?', True)
        return self.log[-1].get_response()

    def get_acceleration(self):
        self.send_command('acceleration?', True)
        return self.log[-1].get_response()
    
    def get_tof(self):
        self.send_command('tof?', True)
        return self.log[-1].get_response()

    def get_wifi(self):
        self.send_command('wifi?', True)
        return self.log[-1].get_response()
    
