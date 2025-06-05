import cv2
import os

# Set the folder path containing images
BATCH_PATH = "/data/user_data/wenhsuac/chenyuzhang/data/rh20t_data"  # Replace with your folder path

scene_names = os.listdir(BATCH_PATH)
scene_names = sorted(scene_names)

for scene_name in scene_names:
    print(scene_name)
    # if scene_name != '24400334_left':
    #     continue
    DATA_PATH = os.path.join(BATCH_PATH, scene_name)
    folder_path = os.path.join(DATA_PATH, 'blends')  # Replace with your folder path
    output_video = os.path.join(DATA_PATH, f'video_{scene_name}.mp4')    # Output video file name


    # Get list of image files (e.g., .jpg, .png)
    images = [img for img in os.listdir(folder_path) if img.endswith((".jpg", ".png"))]
    images.sort(key=lambda x: int(x.split('.')[0]))  # Sort to ensure correct order
    print(images)

    # Read the first image to get dimensions
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 30 FPS

    # Add each image to the video
    for image in images:
        image_path = os.path.join(folder_path, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write frame to video

    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")