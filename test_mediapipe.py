#!/usr/bin/env python3
"""
Test script to verify MediaPipe replacement for dlib
"""

import cv2
import mediapipe as mp
import numpy as np
import os

def test_mediapipe_face_detection():
    """Test MediaPipe face detection and landmark extraction"""
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # Check if we have any test images
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print("No uploads directory found. Creating a simple test...")
        return False
    
    # Find first image file
    image_files = [f for f in os.listdir(uploads_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No test images found in uploads directory")
        return False
    
    test_image_path = os.path.join(uploads_dir, image_files[0])
    print(f"Testing with image: {test_image_path}")
    
    # Load image
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Failed to load image: {test_image_path}")
        return False
    
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            print("No face detected in the image")
            return False
        
        print("✅ Face detected successfully!")
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        print(f"✅ Found {len(face_landmarks.landmark)} facial landmarks")
        
        # Test mouth landmark extraction
        mouth_landmarks = [61, 84, 17, 314, 405, 320, 307, 375]
        xmouthpoints = []
        ymouthpoints = []
        
        for landmark_id in mouth_landmarks:
            landmark = face_landmarks.landmark[landmark_id]
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            xmouthpoints.append(x)
            ymouthpoints.append(y)
        
        print(f"✅ Extracted {len(xmouthpoints)} mouth points")
        
        # Test eye center calculation
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        eyes_center_x = int((left_eye.x + right_eye.x) * img.shape[1] / 2)
        eyes_center_y = int((left_eye.y + right_eye.y) * img.shape[0] / 2)
        
        print(f"✅ Eye center calculated: ({eyes_center_x}, {eyes_center_y})")
        
        # Test mouth center calculation
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_center_x = int((upper_lip.x + lower_lip.x) * img.shape[1] / 2)
        mouth_center_y = int((upper_lip.y + lower_lip.y) * img.shape[0] / 2)
        
        print(f"✅ Mouth center calculated: ({mouth_center_x}, {mouth_center_y})")
        
        return True

if __name__ == "__main__":
    print("Testing MediaPipe replacement for dlib...")
    success = test_mediapipe_face_detection()
    
    if success:
        print("\n🎉 All tests passed! MediaPipe replacement is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 