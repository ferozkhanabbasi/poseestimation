import streamlit as st
import cv2
import tempfile
from PIL import Image
import pandas as pd
import mediapipe as mp
import time
import pickle
import matplotlib.pyplot as plt


from functions import findDistance
from functions import plot_world_landmarks

image = Image.open('logo.png')
st.sidebar.image(image, width=250)
st.sidebar.title("QUANTIFYING GOLF WITH AI")


FrameCount=0
f = st.sidebar.file_uploader("Upload Golf Video")

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())
    vf = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    figure_3D = st.empty()

    FramesVideo = int(vf.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames inside video
    FrameCount = 0 # Currently playing frame
    prevTime = 0
    # some objects for mediapipe
    mpPose = mp.solutions.pose
    mpDraw = mp.solutions.drawing_utils
    pose = mpPose.Pose()


    while vf.isOpened():
      
        FrameCount += 1
        #read image and convert to rgb
        success, img = vf.read()
        if not success:
            print("Can't receive frame. Exiting ...")
            break
        img = cv2.resize(img, (800, 600)) #resizing frame
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #process image
        results = pose.process(image=imgRGB)
        #print("RESULTS: ", type(results.pose_landmarks), results.pose_landmarks)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            #get landmark positions
            landmarks = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape 
                cx, cy = int(lm.x * w), int(lm.y * h) 
                cv2.putText(img, str(id), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)

        # calculate and print fps
        frameTime = time.time()
        fps = 1/(frameTime-prevTime)
        prevTime = frameTime
        
        #cv2.putText(img, "text", (30,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
        #st.write("Pushups : ", pushup_count)

        #pushup_counter.markdown("**PUSHUPS**: %s" % str(pushup_count)+"              "+ "**SQUATS**: %s" % str(squat_count))
        #squat_counter.markdown("**SQUATS**: %s" % str(squat_count))

        #img_ret

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
        if results.pose_landmarks is not None:
            img_ret = plot_world_landmarks(plt, ax, fig,results.pose_landmarks)
            

        # ----------- merge 2 videos ------------
        #img_final = cv2.resize(img, None, fx=1, fy=1)
        #img_final[-img_ret.shape[0]:, -img_ret.shape[1]:] = img_ret
        


        stframe.image(img)
        figure_3D.pyplot(img_ret)