#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 10:56:20 2025

@author: Blyss Sarania
License: MIT
"""
import argparse
import os
import subprocess
import math


def main():
    parser = argparse.ArgumentParser(description="Split a video into segments with N frames at X frame rate.")
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--framerate", type=float, required=True, help="Desired frame rate (X).")
    parser.add_argument("--frames", type=int, required=True, help="Number of frames (N) per segment.")
    parser.add_argument("--offset", type=float, required=False, default=0, help="Offset the segments to help isolate the one you want")
    parser.add_argument("--cpu", required=False, action="store_true", help="Run encoding on CPU instead of GPU(Useful if GPU has issues or not available)")
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output
    target_fps = args.framerate
    frames_per_segment = args.frames

    os.makedirs(output_dir, exist_ok=True)

    cmd_ffprobe = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        input_file
    ]
    try:
        total_frames_str = subprocess.check_output(cmd_ffprobe, universal_newlines=True).strip()
        total_frames = int(total_frames_str)
    except subprocess.CalledProcessError as e:
        print("Error using ffprobe to get total frames:", e)
        return
    except ValueError:
        print("Could not parse total frame count.")
        return

    segment_duration_sec = frames_per_segment / target_fps
    num_segments = math.ceil(total_frames / frames_per_segment)

    for i in range(num_segments):
        start_time_sec = i * segment_duration_sec

        # Name each segment file sequentially
        # e.g., segment_0001.mp4, segment_0002.mp4, etc.
        output_filename = os.path.join(output_dir, f"segment_{i:04d}.mp4")

        # Use ffmpeg to extract each chunk.
        threads = os.cpu_count()
        enc = "-c:v hevc_nvenc -tune hq -cq 16 -preset slow -an" if not args.cpu else f"-c:v libx265 -crf 12 -preset medium -an -threads {threads}"
        cmd_ffmpeg = f"ffmpeg -y -ss {str(start_time_sec + args.offset)} -i '{input_file}' -t  {str(segment_duration_sec)} -r {str(target_fps)} {enc} '{output_filename}'"
        print(cmd_ffmpeg)
        try:
            subprocess.run(cmd_ffmpeg, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing segment {i}:", e)
            break

        print(f"Segment {i} created at {output_filename}")

    print("All segments have been processed.")


if __name__ == "__main__":
    main()
