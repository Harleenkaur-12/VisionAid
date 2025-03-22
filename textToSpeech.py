import pyttsx3
from ultralytics import YOLO

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Initialize YOLOv8n model
        self.model = YOLO('yolov8n.pt')
        
    def read_text(self, frame, speak=True):
        # Process frame with YOLOv8n
        results = self.model(frame)
        
        detected_objects = []
        # Extract detected objects
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                confidence = float(box.conf[0])
                if confidence > 0.5:  # Filter low confidence detections
                    detected_objects.append(class_name)
        
        if speak and detected_objects:
            # Create speech output
            text = "I can see " + ", ".join(list(set(detected_objects)))
            self.engine.say(text)
            self.engine.runAndWait()
            
        return detected_objects
