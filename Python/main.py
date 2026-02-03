# hand_brightness_control.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import serial
import time
import numpy as np
from math import sqrt

# Callback for getting detection results
def result_callback(result: vision.HandLandmarkerResult, output_image: vision.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

latest_result = None

class HandBrightnessController:
    def __init__(self, arduino_port='COM3', baud_rate=9600):
        """
        Initialize hand brightness controller with visual feedback
        """
        # Initialize MediaPipe HandLandmarker using the new Tasks API
        base_options = mp_tasks.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=result_callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Initialize serial connection to Arduino
        self.arduino = None
        try:
            self.arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
            time.sleep(2)  # Wait for connection to establish
            print(f"✅ Connected to Arduino on {arduino_port}")
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            print("Running in simulation mode...")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Could not open webcam")
        
        # Brightness control parameters
        self.min_brightness = 0
        self.max_brightness = 255
        self.current_brightness = 0
        
        # For smoothing brightness values
        self.brightness_history = []
        self.history_size = 5
        
        print("🎬 Camera initialized successfully")
    
    def calculate_hand_openness(self, landmarks):
        """
        Calculate how open the hand is (0 = closed fist, 1 = fully open)
        """
        # Key landmarks for finger tips and bases
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        bases = [2, 5, 9, 13, 17]  # Corresponding base joints
        
        openness_scores = []
        
        for tip, base in zip(tips, bases):
            # Calculate distance between tip and base (normalized coordinates)
            dx = landmarks[tip].x - landmarks[base].x
            dy = landmarks[tip].y - landmarks[base].y
            distance = sqrt(dx*dx + dy*dy)
            openness_scores.append(distance)
        
        # Normalize and average the scores
        avg_openness = sum(openness_scores) / len(openness_scores)
        
        # Apply some scaling to get better range (adjust these values as needed)
        min_distance = 0.02  # Minimum expected distance for closed hand
        max_distance = 0.25  # Maximum expected distance for open hand
        
        # Clamp and normalize to 0-1 range
        normalized = (avg_openness - min_distance) / (max_distance - min_distance)
        normalized = max(0, min(1, normalized))  # Clamp to 0-1
        
        return normalized
    
    def map_to_brightness(self, openness):
        """
        Map hand openness (0-1) to LED brightness (0-255)
        """
        # Use exponential mapping for better control (feels more natural)
        brightness = int(self.min_brightness + 
                        openness * openness * (self.max_brightness - self.min_brightness))
        
        return brightness
    
    def smooth_brightness(self, new_brightness):
        """
        Apply smoothing to avoid flickering
        """
        self.brightness_history.append(new_brightness)
        if len(self.brightness_history) > self.history_size:
            self.brightness_history.pop(0)
        
        # Use weighted average (more weight to recent values)
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][:len(self.brightness_history)]
        weights = [w/sum(weights) for w in weights]  # Normalize
        
        smoothed = sum(b * w for b, w in zip(self.brightness_history, weights))
        return int(smoothed)
    
    def send_brightness_to_arduino(self, brightness):
        """
        Send brightness value to Arduino
        """
        if self.arduino and self.arduino.is_open:
            try:
                # Send brightness as string ending with newline
                command = f"{brightness}\n"
                self.arduino.write(command.encode())
                
                # Try to read response (optional)
                if self.arduino.in_waiting:
                    response = self.arduino.readline().decode().strip()
                    if response:
                        print(f"Arduino: {response}")
            except Exception as e:
                print(f"Serial write error: {e}")
    
    def draw_custom_landmarks(self, image, landmarks):
        """
        Draw hand landmarks with custom styling
        """
        h, w, c = image.shape
        
        # Define hand connections manually (21 landmarks with their connections)
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections (black lines) - thicker and more visible
        for connection in hand_connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = (int(landmarks[start_idx].x * w), 
                          int(landmarks[start_idx].y * h))
            end_point = (int(landmarks[end_idx].x * w), 
                        int(landmarks[end_idx].y * h))
            
            # Draw thick black line
            cv2.line(image, start_point, end_point, (0, 0, 0), 3)
        
        # Draw landmarks (black dots with white outline for better visibility)
        for idx, landmark in enumerate(landmarks):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # Draw white outer circle (for contrast)
            cv2.circle(image, (cx, cy), 8, (255, 255, 255), -1)
            # Draw black inner circle
            cv2.circle(image, (cx, cy), 6, (0, 0, 0), -1)
        
        return image
    
    def draw_brightness_display(self, image, openness, brightness):
        """
        Draw visual feedback for brightness control
        """
        h, w, _ = image.shape
        
        # Create semi-transparent overlay for info panel
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (50, 50, 50), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Draw brightness bar
        bar_width = 200
        bar_height = 30
        bar_x, bar_y = 50, 50
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Filled bar based on brightness
        fill_width = int((brightness / 255) * bar_width)
        # Color changes from red (low) to yellow (high)
        color = (0, int(brightness * 0.7), brightness)
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Border for bar
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Text info
        cv2.putText(image, f"Hand Openness: {openness:.2f}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Brightness: {brightness}/255", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Draw hand openness visual indicator
        openness_radius = int(openness * 40) + 10
        cv2.circle(image, (w - 60, 60), openness_radius, 
                  (0, int(255 * openness), 255), -1)
        cv2.putText(image, "Hand", (w - 80, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def run(self):
        """
        Main loop for hand brightness control
        """
        global latest_result
        
        print("\n" + "="*50)
        print("🤚 HAND BRIGHTNESS CONTROLLER")
        print("="*50)
        print("Instructions:")
        print("1. Show your hand to the camera")
        print("2. Make a FIST → LED OFF (0 brightness)")
        print("3. OPEN hand fully → LED MAX (255 brightness)")
        print("4. Partially open → Control brightness")
        print("5. Press 'q' to quit")
        print("="*50 + "\n")
        
        frame_count = 0
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process the frame for hand detection
            self.detector.detect_async(mp_image, frame_count * 30)
            
            hand_detected = False
            openness = 0
            
            if latest_result and latest_result.hand_landmarks:
                hand_detected = True
                
                for hand_landmarks in latest_result.hand_landmarks:
                    # Draw custom landmarks with black dots and lines
                    frame = self.draw_custom_landmarks(frame, hand_landmarks)
                    
                    # Calculate hand openness
                    openness = self.calculate_hand_openness(hand_landmarks)
                    
                    # Map to brightness and apply smoothing
                    raw_brightness = self.map_to_brightness(openness)
                    smoothed_brightness = self.smooth_brightness(raw_brightness)
                    
                    # Update current brightness if changed significantly
                    if abs(smoothed_brightness - self.current_brightness) > 2:
                        self.current_brightness = smoothed_brightness
                        self.send_brightness_to_arduino(self.current_brightness)
            
            else:
                # No hand detected, fade out brightness
                if self.current_brightness > 0:
                    self.current_brightness = max(0, self.current_brightness - 10)
                    self.send_brightness_to_arduino(self.current_brightness)
            
            # Draw brightness display
            frame = self.draw_brightness_display(frame, openness, self.current_brightness)
            
            # Add status indicator
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
            cv2.putText(frame, status_text, (w - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Display the frame
            cv2.imshow('Hand Brightness Control', frame)
            
            # Exit on 'q' press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset brightness
                self.current_brightness = 0
                self.send_brightness_to_arduino(0)
                print("Reset brightness to 0")
            elif key == ord('m'):  # Max brightness
                self.current_brightness = 255
                self.send_brightness_to_arduino(255)
                print("Set brightness to max (255)")
            
            frame_count += 1
        
        self.cleanup()
    
    def cleanup(self):
        """
        Release resources
        """
        print("\n" + "="*50)
        print("Cleaning up...")
        
        # Turn off LED before exiting
        if self.arduino and self.arduino.is_open:
            self.send_brightness_to_arduino(0)
            time.sleep(0.1)
            self.arduino.close()
            print("Arduino connection closed")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released")
        print("Program terminated successfully!")
        print("="*50)

# Main execution with error handling
if __name__ == "__main__":
    try:
        # Use macOS port
        controller = HandBrightnessController(arduino_port='/dev/cu.usbserial-140')
        controller.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check Arduino is connected to correct USB port")
        print("2. Install CH340 drivers if needed")
        print("3. Check webcam is working")
        print("4. Run: pip install opencv-python mediapipe pyserial")

 