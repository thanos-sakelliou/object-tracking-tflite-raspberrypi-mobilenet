from imutils.video import FPS
import numpy as np
import time
import cv2

class PiCameraStream(object):

    def __init__(self,
                 resolution=[320, 240],
                 framerate=24,
                 ):

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
        # Init resolution
        self.resolution = tuple(resolution)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Init framerate
        self.framerate = framerate
        fps = FPS().start()

        # Init stop variable
        self.stop = False
        
        while self.stop == False :
            
            # Capture frame-by-frame
            read, self.frame = cap.read()
            
            # Check if the image is read correctly (read is bool)
            if not read:
                print("Can't read the image ...")
                break

            # Display the resulting frame
            cv2.imshow('frame', self.frame)
            
            # Break loop with q
            if cv2.waitKey(1) == ord('q'):
                break
            
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        

    def render_overlay(self):
        print("render_overlay not defined")

    def start_overlay(self):
        print('start_overlay not defined')

    def start(self):
        return self

    def flush(self):
        return None

    def read(self):
        return self.frame

    def stop(self):
        self.stop = True