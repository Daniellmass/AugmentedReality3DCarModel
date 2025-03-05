import cv2
import numpy as np
import trimesh
import pyrender
import math
import glob
import os


# ======= CONSTANTS ========

TEMPLATE_IMAGE_PATH = "temp_pic.jpeg"         
VIDEO_INPUT_PATH = "input_vid.mp4"            
VIDEO_OUTPUT_PATH = "new_output_video.mp4"         
CALIBRATION_IMAGES_PATH = "./calib_dir/*.jpeg"   
CHESSBOARD_SIZE = (7, 7)                        
SQUARE_SIZE = 2.88                             
CAR_PATH = "models/Car.obj"                         

# ===== CAMERA CALIBRATION ==

calibration_images = glob.glob(CALIBRATION_IMAGES_PATH)
obj_points = []  
img_points = []  

pattern_points = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
pattern_points[:, :2] = np.indices(CHESSBOARD_SIZE).T.reshape(-1, 2)
pattern_points *= SQUARE_SIZE

for img_path in calibration_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE)
    if found:
        img_points.append(corners)
        obj_points.append(pattern_points)
    else:
        print(f"Chessboard not found in {img_path}")

# Calibrate the camera to get the intrinsic matrix and distortion coefficients.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img.shape[::-1], None, None
)
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)


# ======= GOURD CLASS ======

class MyObj:
    def __init__(self, K, width, height, scale_factor=1):
        # Store camera intrinsics and renderer size.
        self.K = K
        self.width = width
        self.height = height

        # Load the mesh from the file.
        self.mesh = trimesh.load(CAR_PATH,process=False)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(list(self.mesh.geometry.values()))
            
        self.mesh.rezero()

        print(self.mesh.visual)
        if hasattr(self.mesh.visual, 'material'):
            print("Material loaded:", self.mesh.visual.material)
        else:
            print("No material information available.")
 
        # Original rotation to stand up the car.
        T = np.eye(4)
        T[0:3, 0:3] = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0]])
        self.mesh.apply_transform(T)

        # Additional rotation (for example, rotate by 90 degrees about Z-axis).
        import math
        angle = math.radians(180)  # adjust angle as needed
        R_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [0,                0,               1]
        ])
        T_rot = np.eye(4)
        T_rot[0:3, 0:3] = R_z
        self.mesh.apply_transform(T_rot)

       # Additional rotation: 270 degrees about the y-axis.
        angle_y = math.radians(180)
        R_y = np.array([
            [math.cos(angle_y), 0, math.sin(angle_y)],
            [0, 1, 0],
            [-math.sin(angle_y), 0, math.cos(angle_y)]
        ])
        T_rot_y = np.eye(4)
        T_rot_y[0:3, 0:3] = R_y
        self.mesh.apply_transform(T_rot_y)

        # New transformation: Add 90 degrees about the x-axis ---
        angle_x = math.radians(90)
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(angle_x), -math.sin(angle_x)],
            [0, math.sin(angle_x),  math.cos(angle_x)]
        ])
        T_rot_x = np.eye(4)
        T_rot_x[0:3, 0:3] = R_x
        self.mesh.apply_transform(T_rot_x)

        # Scale the mesh (smaller scale)
        bounds = self.mesh.bounds
        max_bound = np.max(bounds[1] - bounds[0])
        scale = 2 * scale_factor / max_bound  # Adjust the constant (e.g., 5) for further scaling down.
        T_scale = np.eye(4)
        T_scale[0:3, 0:3] = scale * np.eye(3) * 0.98
        self.mesh.apply_transform(T_scale)

        # Convert to pyrender mesh.
        self.mesh = pyrender.Mesh.from_trimesh(self.mesh)
        
        self.scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
        self.scene.add(self.mesh)
        self.camera = pyrender.IntrinsicsCamera(
            self.K[0, 0],
            self.K[1, 1],
            self.K[0, 2],
            self.K[1, 2],
            znear=0.1,
            zfar=1000,
        )
        self.cam_node = self.scene.add(self.camera)
        light = pyrender.SpotLight(color=np.ones(3), intensity=1000, innerConeAngle=np.pi / 16)
        pose = np.eye(4)
        pose[0:3, 3] = np.array([0, 10, 0])
        self.scene.add(light, pose=pose)
        self.r = pyrender.OffscreenRenderer(self.width, self.height)

        # Existing spotlight (already in your code)
        spot_light = pyrender.SpotLight(color=np.ones(3), intensity=1000, innerConeAngle=np.pi/16)
        spot_pose = np.eye(4)
        spot_pose[0:3, 3] = np.array([0, 10, 0])
        self.scene.add(spot_light, pose=spot_pose)

        # Additional directional light for overall fill:
        directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        directional_pose = np.eye(4)
        # Position or orient the light as needed. For example, coming from top left:
        directional_pose[0:3, 3] = np.array([-5, 5, 5])
        self.scene.add(directional_light, pose=directional_pose)

        # Additional point light to further fill shadows:
        point_light = pyrender.PointLight(color=np.ones(3), intensity=500)
        point_pose = np.eye(4)
        # Position the point light at a different angle:
        point_pose[0:3, 3] = np.array([5, 5, -5])
        self.scene.add(point_light, pose=point_pose)

    def draw(self, img, rvec, tvec):
        
        pose = np.eye(4)
        # Convert rotation vector to rotation matrix.
        res_R, _ = cv2.Rodrigues(rvec)
        # Set rotation: note the use of transpose.
        pose[0:3, 0:3] = res_R.T
        # Set translation: the negative product aligns with the camera convention.
        pose[0:3, 3] = (-res_R.T @ tvec).flatten()
        # Flip the y and z axes so that the rendered object matches the scene orientation.
        flip = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])
        pose = pose @ flip

        self.scene.set_pose(self.cam_node, pose)

        # Render the scene.
        color, depth = self.r.render(self.scene)

        cv2.imwrite("debug_render.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Create a binary mask from the depth map.
        mask = (depth > 0).astype(np.uint8) * 255

        # Convert the rendered image from RGB (pyrender) to BGR (OpenCV).
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        mask_inv = cv2.bitwise_not(mask)

        # Resize the background image to match the renderer’s dimensions.
        img = cv2.resize(img, (self.width, self.height))
        # Blend the rendered object with the background.
        img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        gourd_fg = cv2.bitwise_and(color, color, mask=mask)
        blended = cv2.add(img_bg, gourd_fg)
        return blended



# === VIDEO PROCESSING =====
def process_video_with_3D_car(
    template_img_path,
    video_input_path,
    video_output_path,
    camera_matrix,
    dist_coeffs,
    square_size,
    gourd_instance,
):
    # Initialize SIFT and compute descriptors for the template image.
    sift = cv2.SIFT_create()
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    template_kp, template_des = sift.detectAndCompute(template_img, None)

    # Open the input video.
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties.
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    bf = cv2.BFMatcher()

    # Animation parameters (to move the car along the X-axis).
    car_offset = 0.0      
    offset_speed = 0.05   
    offset_min, offset_max = 0, 2.0  
    direction = +1

    cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cnt +=1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = sift.detectAndCompute(gray_frame, None)

        if frame_des is None or len(frame_des) < 2:
            out.write(frame)
            continue

        matches = bf.knnMatch(template_des, frame_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) > 10:
            src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                inliers_mask = (mask.ravel() == 1)
                src_inliers = src_pts[inliers_mask]
                dst_inliers = dst_pts[inliers_mask].reshape(-1, 2)

                # Build corresponding 3D points (assume the template lies on z=0).
                xyz_template = []
                for p in src_inliers[:, 0]:
                    x_3d = p[0] * (square_size / template_img.shape[1])
                    y_3d = p[1] * (square_size / template_img.shape[0])
                    xyz_template.append([x_3d, y_3d, 0.0])
                xyz_template = np.array(xyz_template, dtype=np.float32)

                # Solve for the camera pose using solvePnP.
                retval, r_vec, t_vec = cv2.solvePnP(
                    xyz_template,
                    dst_inliers,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if retval:
                    # Compute the car's forward direction from its rotation.
                    R, _ = cv2.Rodrigues(r_vec)
                    # Define the car's local forward direction (assumed to be the x-axis).
                    forward = R @ np.array([1, 0, 0], dtype=np.float32)
                    
                    # Update the translation vector: move the car along its forward direction.
                    t_vec_anim = t_vec.copy() + car_offset * forward.reshape(3, 1)
                    t_vec_anim[2] += 1.0                    
                    # Render the car using the updated pose.
                    frame = gourd_instance.draw(frame, r_vec, t_vec_anim)


        # Update the animation offset.
        car_offset += direction * offset_speed
        if car_offset > offset_max:
            car_offset = offset_max
            direction = -1
        elif car_offset < offset_min:
            car_offset = offset_min
            direction = +1

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to:", video_output_path)


# === SETUP & EXECUTION ====
cap_temp = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap_temp.isOpened():
    raise IOError("Could not open video file for reading dimensions.")
frame_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_temp.release()

# Create an instance of the Gourd class.
gourd_instance = MyObj(camera_matrix, frame_width, frame_height, scale_factor=1)

process_video_with_3D_car(
    TEMPLATE_IMAGE_PATH,
    VIDEO_INPUT_PATH,
    VIDEO_OUTPUT_PATH,
    camera_matrix,
    dist_coeffs,
    SQUARE_SIZE,
    gourd_instance,
)
