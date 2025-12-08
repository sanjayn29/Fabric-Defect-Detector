# fabric_defect_detector_pi.py
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os

# Constants
MODEL_PATH = 'fabric_defect_model_quant.tflite'

CLASS_NAMES = [
    'broken_stitch', 'defect-free', 'hole',
    'horizontal', 'lines', 'needle_mark',
    'pinched_fabric', 'stain', 'vertical'
]
INPUT_SIZE = (224, 224)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFECT_THRESHOLD = 0.85  # Only classify as defect if confidence > 85%

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    """Preprocess for TFLite model"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize(INPUT_SIZE)
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def predict_with_tta(frame, n_augment=3):
    """Test-Time Augmentation for better accuracy"""
    original = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], original)
    interpreter.invoke()
    base_pred = interpreter.get_tensor(output_details[0]['index'])
    
    # Augmentations
    h_flip = cv2.flip(frame, 1)
    v_flip = cv2.flip(frame, 0)
    
    preds = [base_pred]
    for aug_img in [h_flip, v_flip]:
        processed = preprocess_image(aug_img)
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        preds.append(interpreter.get_tensor(output_details[0]['index']))
    
    avg_pred = np.mean(preds, axis=0)
    pred_class = CLASS_NAMES[np.argmax(avg_pred)]
    confidence = np.max(avg_pred)
    
    return pred_class, confidence

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return
    
    print("Fabric Defect Detector Running...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed")
            break
        
        start_time = time.time()
        
        # Predict with TTA
        pred_class, confidence = predict_with_tta(frame)
        inference_time = time.time() - start_time
        
        # Display
        display_text = f"{pred_class} ({confidence:.2f})"
        color = (0, 255, 0) if pred_class == 'defect-free' else (0, 0, 255)
        
        cv2.putText(frame, display_text, (20, 40), FONT, 0.9, color, 2)
        cv2.putText(frame, f"FPS: {1/inference_time:.1f}", (20, 80), FONT, 0.7, (255, 255, 255), 2)
        
        # Highlight defects above threshold
        if pred_class != 'defect-free' and confidence > DEFECT_THRESHOLD:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            cv2.putText(frame, "DEFECT DETECTED!", (frame.shape[1]//4, 120), 
                        FONT, 1.5, (0, 0, 255), 3)
        
        cv2.imshow('Fabric Defect Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
