# ======= imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ======= constants
SQUARE_SIZE = 2.88
  
# ===== video input, output and metadata
TEMPLATE_IMAGE_PATH = "temp_pic.jpeg"  
VIDEO_INPUT_PATH = "input_video.mp4"
VIDEO_OUTPUT_PATH = "output_video.mp4"

def process_video(template_img_path, video_input_path, video_output_path):
    # Load the template image and compute its keypoints and descriptors
    sift = cv2.SIFT_create()
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    template_kp, template_des = sift.detectAndCompute(template_img, None)

    # Open the video input
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize BFMatcher
    bf = cv2.BFMatcher()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in the current frame
        frame_kp, frame_des = sift.detectAndCompute(gray_frame, None)

        # Match descriptors between the template and the frame
        matches = bf.knnMatch(template_des, frame_des, k=2)

        # Apply ratio test to filter good matches
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # If there are enough good matches, compute the homography
        if len(good_matches) > 5:
            src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            distances = np.linalg.norm(src_pts - dst_pts, axis=2)
            avg_distance = np.mean(distances)
        
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
           
            if H is not None:
                # Warp the template image onto the video frame
                h, w = template_img.shape
                warped_img = cv2.warpPerspective(template_img, H, (frame.shape[1], frame.shape[0]))

                # Blend the warped image with the original frame
                warped_img_colored = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)  # Match color channels
                blended_frame = cv2.addWeighted(frame, 0.5, warped_img_colored, 0.5, 0)
            else:
                print("Homography computation failed. Writing original frame.")
                blended_frame = frame
        else:
            print("Not enough good matches. Writing original frame.")
            blended_frame = frame

        # Write the blended frame to the output video
        out.write(blended_frame)

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to:", video_output_path)

process_video(TEMPLATE_IMAGE_PATH, VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH)

