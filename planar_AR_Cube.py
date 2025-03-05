# ======= imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ======= constants
TEMPLATE_IMAGE_PATH = "temp_pic.jpeg"  
VIDEO_INPUT_PATH = "input_video.mp4"
VIDEO_OUTPUT_PATH = "output_video.mp4"
CALIBRATION_IMAGES_PATH = "./calibration_images/*.jpeg"
CHESSBOARD_SIZE = (7, 7)  
SQUARE_SIZE = 2.88  

# ===== Camera calibration
# 1. Collect calibration images (e.g., chessboard patterns)
calibration_images = glob(CALIBRATION_IMAGES_PATH)
obj_points = []  # 3D points in real-world space
img_points = []  # 2D points in image plane

# Prepare a 3D chessboard grid
pattern_points = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
pattern_points[:, :2] = np.indices(CHESSBOARD_SIZE).T.reshape(-1, 2)
pattern_points *= SQUARE_SIZE

# Detect corners in all calibration images
for img_path in calibration_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE)
    if found:
        img_points.append(corners)
        obj_points.append(pattern_points)

        # Debug: Visualize detected corners
        img_with_corners = cv2.drawChessboardCorners(img.copy(), CHESSBOARD_SIZE, corners, found)
        plt.figure(figsize=(8, 6))
        plt.imshow(img_with_corners, cmap="gray")
        plt.title(f"Chessboard Found: {img_path}")
        plt.show()
    else:
        print(f"Chessboard not found in {img_path}")

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img.shape[::-1], None, None
)
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# ===== Define cube's 3D points in real-world coordinates
objectPoints = (
    SQUARE_SIZE * np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1],  # Top face
        ]
    )
)

# Helper function to draw the cube
def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # Draw pillars in blue
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # Draw top face in red
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

# ===== Process video and overlay cube
def process_video(template_img_path, video_input_path, video_output_path, camera_matrix, dist_coeffs):
    # Load the template image and compute its keypoints and descriptors
    sift = cv2.SIFT_create()
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    template_kp, template_des = sift.detectAndCompute(template_img, None)

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

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
        good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

        # If there are enough good matches, compute the homography
        if len(good_matches) > 10:
            src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Subset of keypoints that obey homography
                src_pts_inliers = src_pts[mask.ravel() == 1]
                dst_pts_inliers = dst_pts[mask.ravel() == 1]

                # Solve PnP to get r_vec and t_vec
                xyz_template = np.float32([[p[0] * SQUARE_SIZE / template_img.shape[1], 
                                             p[1] * SQUARE_SIZE / template_img.shape[0], 
                                             0] for p in src_pts_inliers[:, 0]])

                _, r_vec, t_vec = cv2.solvePnP(
                    xyz_template, dst_pts_inliers, camera_matrix, dist_coeffs
                )

                # Project 3D points to 2D
                imgpts, _ = cv2.projectPoints(objectPoints, r_vec, t_vec, camera_matrix, dist_coeffs)
                frame = draw_cube(frame, imgpts)

        # Write the blended frame to the output video
        out.write(frame)

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to:", video_output_path)

# Run the process
process_video(TEMPLATE_IMAGE_PATH, VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH, camera_matrix, dist_coeffs)
