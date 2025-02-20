import supervision as sv
from ultralytics import YOLO 
import cv2
import numpy as np

def initialize_model():
    """Initialize the YOLO model with the fixed weights path."""
    weights_path = "/home/cma/hucou/LightGlue/SuperGluePretrainedNetwork/yolov8n.pt"
    return YOLO(weights_path)

def initialize_trackers():
    """Initialize all necessary tracking and annotation components."""
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Define counting line
    line_start = sv.Point(0, 300)
    line_end = sv.Point(640, 300)
    line_counter = sv.LineZone(start=line_start, end=line_end)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)
    
    return tracker, box_annotator, label_annotator, line_counter, line_annotator

def process_frame(frame, model, tracker, box_annotator, label_annotator, line_counter, line_annotator):
    """Process a single frame and return the annotated frame."""
    # Model inference
    results = model(frame, verbose=False, conf=0.3, iou=0.7)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter for persons only (class_id 0)
    detections = detections[np.where(detections.class_id == 0)]
    detections = tracker.update_with_detections(detections)
    
    # Annotate frame
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    
    # Create labels
    labels = [f"Person #{tracker_id} {conf:.2f}" 
             for tracker_id, conf in zip(detections.tracker_id, detections.confidence)]
    
    # Update counter and annotate
    line_counter.trigger(detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
    
    # Add total count
    cv2.putText(
        annotated_frame,
        f"Total Count: {line_counter.in_count + line_counter.out_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    return annotated_frame

def main():
    """Main function to run the human detection and tracking system."""
    # Initialize components
    model = initialize_model()
    tracker, box_annotator, label_annotator, line_counter, line_annotator = initialize_trackers()
    
    # Initialize video capture (0 for default webcam)
    video_path = "/home/cma/hucou/t6.mp4"
    cap = cv2.VideoCapture(video_path)
    
    print("Starting human detection and tracking...")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
                
            # Process and display frame
            annotated_frame = process_frame(
                frame, 
                model, 
                tracker, 
                box_annotator, 
                label_annotator, 
                line_counter, 
                line_annotator
            )
            
            cv2.imshow('Human Detection and Tracking', annotated_frame)
        
            # Check for quit command
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Quitting...")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == "__main__":
    main()