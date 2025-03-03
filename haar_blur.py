#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 10:56:20 2025

@author: blyss
"""
import subprocess
import argparse
import cv2
import os
import shutil
os.makedirs("frames", exist_ok=True)
frame_count = 0
parser = argparse.ArgumentParser(description="Blurs faces using Haar cascade classification.")
parser.add_argument("--input", required=True, help="Path to the input video file.")
parser.add_argument("--output", required=True, help="Path to the output video file.")
parser.add_argument("--classifier", required=True, help="Path to haarcascade_frontalface_default.xml")
args = parser.parse_args()
face_cascade = args.classifier
cap = cv2.VideoCapture(args.input)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames or can't read from source.")
        break
    print(f"Processing frame {frame_count}: {frame.shape}...")
    # Convert frame to grayscale for the face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Loop through detected faces and blur or pixelate
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        # Apply a blur
        face_roi = cv2.GaussianBlur(face_roi, (41, 41), 30)
        frame[y:y + h, x:x + w] = face_roi

    # Write the censored frame for later FFMPEGing
    cv2.imwrite(f"frames/frame_{frame_count:05d}.png", frame)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
threads = os.cpu_count()
ffmpeg_cmd = f"ffmpeg -framerate 24 -i frames/frame_%05d.png -c:v libx265 -crf 12 -preset slow -b:v 10M  -pix_fmt yuv420p -threads {threads} {args.output}"
print(f"FFMPEGing with {ffmpeg_cmd}...")
try:
    subprocess.run(ffmpeg_cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error encoding frames with ffmpeg: {e}", e)
    exit(4)
shutil.rmtree("./frames")
print("Done!")
