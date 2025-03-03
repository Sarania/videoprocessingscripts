#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 10:56:20 2025

@author: Blyss Sarania
License: MIT
"""

import subprocess
import argparse
import cv2
import os
import shutil
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Blur faces in video using YOLO")
parser.add_argument("--input", required=True, help="Path to the input video file.")
parser.add_argument("--output", required=True, help="Path to save the blurred video.")
parser.add_argument("--strength", required=False, type=int, default=51, help="Strength of blur.")
parser.add_argument("--model", default="yolov8n-face.pt", help="Path to your YOLO face detection model.")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for face detection.")
args = parser.parse_args()
if args.strength % 2 == 0:
    args.strength += 1  # Kernel size must be odd

os.makedirs("frames", exist_ok=True)
frame_count = 0
model = YOLO(args.model)
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

    results = model.predict(source=frame, conf=args.conf, verbose=False)

    if len(results) > 0:
        # We only have one frame in the list, so take the first result
        dets = results[0].boxes
        if dets is not None:
            for box in dets:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                face_roi = frame[y1:y2, x1:x2]  # Blur it
                blurred_roi = cv2.GaussianBlur(face_roi, (args.strength, args.strength), 40)
                frame[y1:y2, x1:x2] = blurred_roi
    cv2.imwrite(f"frames/frame_{frame_count:05d}.png", frame)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
threads = os.cpu_count()
ffmpeg_cmd = (f"ffmpeg -y -framerate {int(fps)} -i frames/frame_%05d.png -c:v libx265 -crf 12 -preset slow -b:v 10M -pix_fmt yuv420p -threads {threads} {args.output}")
print(f"FFMPEGing with {ffmpeg_cmd}...")
try:
    subprocess.run(ffmpeg_cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error encoding frames with ffmpeg: {e}")
    exit(4)

shutil.rmtree("frames")
print("Done!")
