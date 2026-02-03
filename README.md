# ✋ Gesture Controlled LED using Arduino + Python + MediaPipe 💡

Control an LED using **hand gestures** with Arduino and Python!  
This project uses **MediaPipe hand tracking** to detect finger movement and control LED brightness in real time. 🚀

---

## 🎯 Project Features
- *✋* Real-time hand tracking using AI  
- *💡* Gesture-based LED brightness control  
- *🔌* Serial communication between Python & Arduino  
- *⚡* Smooth PWM LED control  

---

## 🔧 Components Required
- Arduino Uno / Nano  
- LED 
- Jumper Wires  
- USB Cable  
- Webcam  
- Python (OpenCV + MediaPipe)  

---

## ⚙️ How It Works
1. Python detects thumb and index finger distance using MediaPipe.  
2. The distance is converted into an angle (0–180).  
3. Angle is sent to Arduino via Serial Communication.  
4. Arduino maps the angle to PWM and controls LED brightness.  

---

## 📂 Project Structure
/Python

/Arduino

README.md


---

## 🚀 Future Improvements
- Add multiple LEDs  
- Control fan or servo motor  
- Wireless control using Bluetooth  
- GUI dashboard for brightness control  

---

✨ Enjoy controlling electronics with your hand gestures!
