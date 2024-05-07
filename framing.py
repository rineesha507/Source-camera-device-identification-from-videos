import os
import cv2

# Function to extract frames from a video file
def extract_frames(video_path, frame_rate, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % frame_rate == 0:
                resized_frame = cv2.resize(frame, target_size)
                frames.append(resized_frame)

            frame_count += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames

# Function to extract frames from all video files in a directory
def extract_frames_from_videos(video_directory, frame_rate, target_size=(224, 224)):
    frames_list = []
    for file in os.listdir(video_directory):
        #print(file)
        if file.endswith('.mp4') or file.endswith('.mov'):
            video_path = os.path.join(video_directory, file)
            print(video_path)
            frames = extract_frames(video_path, frame_rate, target_size)
            frames_list.append(frames)
    #print(frames_list)
    return frames_list

train_dataset_dir = r'C:/Users/rineesha/Documents/Project/train2'
for file in os.listdir(train_dataset_dir):
    video_in_path = os.path.join(train_dataset_dir, file)
    frame_rate = 120
    frames_list = extract_frames_from_videos(video_in_path, frame_rate)
    train_data_dir = r'C:/Users/rineesha/Documents/Project/directory'
    video_out_path = os.path.join(train_data_dir, file)
    os.mkdir(video_out_path)
    x = 0
    for video_frames in frames_list:
        x = x + 1
        for i, frame in enumerate(video_frames):
            file_path = os.path.join(video_out_path, f'frame_{x}_{i}.jpg')
            cv2.imwrite(file_path, frame)

print("done")
