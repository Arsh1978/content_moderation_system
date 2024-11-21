import streamlit as st
import whisper
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time
import tempfile
import os
from moviepy.editor import VideoFileClip
from profanity_check import predict



def extract_audio(video_path, output_path):
    """Extract audio from video file"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()

def check_profanity(audio_path):
    # Transcribe audio using whisper
    model_base = whisper.load_model("base.en")
    result_base = model_base.transcribe(audio_path)
    transcript = result_base["text"]
    
    # Convert transcript to array format for predict
    text_array = [transcript]
    
    # Use ML-based profanity detection
    result = predict(text_array)
    
    return bool(result[0]), "Profanity detected in audio" if result[0] else "No profanity detected"

def check_nudity(frame, nudity_model, target_labels, confidence_threshold=0.50):
    results = nudity_model.predict(source=frame, verbose=False)
    
    for result in results:
        for i in range(len(result.boxes)):
            label = result.names[result.boxes.cls[i].item()]
            confidence = result.boxes.conf[i].item()
            
            if label in target_labels and confidence > confidence_threshold:
                return True
    return False

def check_weapons(frame, weapon_model, confidence_threshold=0.40):
    target_labels = ["gun", "knife"]
    results = weapon_model(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            label = result.names[int(box.cls)]
            confidence = box.conf.item()
            
            if label in target_labels and confidence > confidence_threshold:
                return True
    return False

def check_middle_finger(frame, hands):
    mp_hands = mp.solutions.hands
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if (is_finger_up(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                not is_finger_up(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                not is_finger_up(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP) and
                not is_finger_up(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)):
                return True
    return False

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y

def content_moderation_pipeline(video_path, audio_path, nudity_model_path, weapon_model_path):
    # Initialize models
    nudity_model = YOLO(nudity_model_path)
    weapon_model = YOLO(weapon_model_path)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Check for profanity in audio
    status_text.text("Checking audio for profanity...")
    is_profane, profanity_reason = check_profanity(audio_path)
    if is_profane:
        return True, profanity_reason
    
    # Process video frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    second_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % int(fps) == 0:
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame at {second_count} seconds...")
            
            if check_nudity(frame, nudity_model, [
                "BELLY_EXPOSED", "ANUS_EXPOSED",
                "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
                "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED"
            ]):
                cap.release()
                return True, f"Nudity detected at {second_count} seconds"
            
            if check_weapons(frame, weapon_model):
                cap.release()
                return True, f"Weapon detected at {second_count} seconds"
            
            if check_middle_finger(frame, hands):
                cap.release()
                return True, f"Middle finger detected at {second_count} seconds"
            
            second_count += 1
        
        frame_count += 1
    
    cap.release()
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    return False, "No inappropriate content detected"

def main():
    st.title("Content Moderation System")
    st.write("Upload a video file to check for inappropriate content")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Use context manager for temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            video_path = os.path.join(temp_dir, "temp_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getvalue())  # Use getvalue() for Streamlit file uploader
            
            # Extract audio
            audio_path = os.path.join(temp_dir, "temp_audio.mp3")
            with st.spinner("Extracting audio from video..."):
                extract_audio(video_path, audio_path)
            
            # Process content
            st.write("Processing video content...")
            start_time = time.time()
            
            # Use absolute paths for models
            nudity_model_path = "nudity.pt"
            weapon_model_path = "weapon.pt"
            
            is_inappropriate, reason = content_moderation_pipeline(
                video_path, audio_path, nudity_model_path, weapon_model_path
            )
            
            end_time = time.time()
            
            # Display results
            st.header("Results")
            st.write(f"Content is inappropriate: {is_inappropriate}")
            st.write(f"Reason: {reason}")
            st.write(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()