import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

class SignLanguageRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the Sign Language Recognizer
        
        :param model_path: Path to a pre-trained model (optional)
        """
        self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # If no model path is provided, create a default model
        if model_path is None:
            self.model = self.create_model()
        else:
            self.model = keras.models.load_model(model_path)

    def create_model(self):
        """
        Create a default CNN model for sign language recognition
        
        :return: Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(64, 64, 1)),  # Assuming grayscale images of 64x64 pixels
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def preprocess_frame(self, frame):
        """
        Preprocess the camera frame for model prediction
        
        :param frame: Input camera frame
        :return: Processed image ready for prediction
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, (64, 64))
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Reshape for model input
        input_image = normalized.reshape(1, 64, 64, 1)
        
        return input_image

    def recognize_sign(self, frame):
        """
        Recognize the sign language character in the frame
        
        :param frame: Input camera frame
        :return: Predicted character and confidence
        """
        preprocessed = self.preprocess_frame(frame)
        predictions = self.model.predict(preprocessed)
        
        # Get the index of the highest confidence prediction
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        
        return self.labels[predicted_index], confidence

    def train_model(self, train_data_path, epochs=10, validation_split=0.2):
        """
        Train the model on sign language images
        
        :param train_data_path: Path to training images directory
        :param epochs: Number of training epochs
        :param validation_split: Percentage of data used for validation
        """
        # Load and preprocess training data
        train_images = []
        train_labels = []
        
        for label in self.labels:
            label_path = os.path.join(train_data_path, label)
            if os.path.exists(label_path):
                for image_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, image_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, (64, 64))
                    train_images.append(img_resized / 255.0)
                    
                    # Create one-hot encoded label
                    label_vector = [0] * len(self.labels)
                    label_vector[self.labels.index(label)] = 1
                    train_labels.append(label_vector)
        
        # Convert to numpy arrays
        train_images = np.array(train_images).reshape(-1, 64, 64, 1)
        train_labels = np.array(train_labels)
        
        # Train the model
        self.model.fit(train_images, train_labels, 
                       epochs=epochs, 
                       validation_split=validation_split)
        
        # Save the trained model
        self.model.save('sign_language_model.h5')

    def run_camera_recognition(self):
        """
        Run real-time sign language recognition through camera
        """
        cap = cv2.VideoCapture(0)
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Detect and recognize sign
            try:
                predicted_char, confidence = self.recognize_sign(frame)
                
                # Display prediction on frame
                display_text = f"{predicted_char} (Confidence: {confidence:.2f})"
                cv2.putText(frame, display_text, 
                            (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Recognition error: {e}")
            
            # Display the resulting frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Create recognizer
    recognizer = SignLanguageRecognizer()
    
    # Option to train model first
    # recognizer.train_model('path/to/training/data')
    
    # Run camera recognition
    recognizer.run_camera_recognition()

if __name__ == "__main__":
    main()