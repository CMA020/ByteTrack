import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import time 

from ultralytics import YOLO

# Path to the video file
import os


class Person:
    def __init__(self, bbox, first_center, person_id,inframe,image):
        """
        Initialize a Person object.
        
        Args:
            bbox (tuple): Bounding box coordinates (x, y, w, h)
            first_center (tuple): Center coordinates of first detection
            person_id (int): Unique identifier for the person
        """
        self.bbox = bbox
        self.first_center = first_center
        self.person_id = person_id
        self.image = image  # Initialized as None as specified
        self.tracking = False 
        self.inframe  = inframe 
        self.undetected  = 0 
    def __repr__(self):
        """
        String representation of the Person object.
        """
        return f"Person(id={self.person_id}, bbox={self.bbox}, first_center={self.first_center})"


class PersonList:
    def __init__(self):
        """
        Initialize a list to store Person objects.
        """
        
        self.door = []
        self.persons_inside  = []
        self.persons_leaving = []
        self.temp_list = []
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.det_close = False
        self.person_counter = 0 
        self.det_intersect = False  
        self.iou=0.0
       
        
    def add_person_inside(self, person):
        """
        Add a Person object to the list.
        
        Args:
            person (Person): Person object to be added
        """
        self.persons_inside.append(person)
    def get_bounding_box_center(self, bbox):
        """
        Calculate the center point of a bounding box.
        
        Args:
            bbox (tuple): Bounding box in format (x, y, width, height)
                x, y: top-left corner coordinates
                width, height: dimensions of the box
        
        Returns:
            tuple: (center_x, center_y) coordinates of the bounding box center
        """
        x, y, width, height = bbox
        center_x = x + width // 2
        center_y = y + height // 2
        return (center_x, center_y)
    # def resize_image(self,img, target_height):
    #         aspect_ratio = img.shape[1] / img.shape[0]
    #         target_width = int(target_height * aspect_ratio)
    #         return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    def preprocess_frame(self, frame):
        """Preprocess camera frame for SuperPoint/SuperGlue"""
        # Convert to grayscale if the frame is in color
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        # Convert to tensor
        inp = torch.from_numpy(frame).float().to(self.device)[None, None] / 255.0
        
        return frame, inp


    def calculate_feature_centroid(self,reference_path, query_path):
        """
        Match two images and return their centroids and number of matched points
        
        Args:
            reference_path: Path to the reference image
            query_path: Path to the query image
            visualize: Whether to show the visualization plot
        
        Returns:
            tuple: (centroid0, centroid1, num_matches)
                - centroid0: (x,y) coordinates of reference image centroid
                - centroid1: (x,y) coordinates of query image centroid
                - num_matches: Number of matched keypoints between images
        """
        # Set up device and config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.0025,
                'max_keypoints': 2048
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 40,
                'match_threshold': 0.2,
            }
        }
        
        # Initialize matcher
        matching = Matching(config).eval().to(device)
        
        # Read images
        image0, inp0 = self.preprocess_frame(reference_path)
        image1, inp1 = self.preprocess_frame(query_path)
        
        # Perform matching
        with torch.no_grad():
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        
        # Extract matched keypoints
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        # Calculate centroids and number of matches
        num_matches = len(mkpts0)
        centroid0 = np.mean(mkpts0, axis=0) if num_matches > 0 else None
        centroid1 = np.mean(mkpts1, axis=0) if num_matches > 0 else None
        
        # Visualize if requested
        
        
        return centroid0, centroid1, num_matches

    def calculate_iou(self, box1, box2):
        """
        Calculate intersection over union between two bounding boxes.
        
        Args:
            box1: tuple (x1, y1, w1, h1)
            box2: tuple (x2, y2, w2, h2)
        
        Returns:
            float: IOU score
        """
        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1, w2, h2 = box2
        x1_2 = x1_1 + w1
        y1_2 = y1_1 + h1
        x2_2 = x2_1 + w2
        y2_2 = y2_1 + h2
        
        # Calculate intersection coordinates
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IOU
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
    # def remove_duplicates_iou(self,frame):
    #     """
    #     Remove duplicate detections based on multiple criteria:
    #     1. First, remove all boxes with width less than 90
    #     2. Then, for remaining boxes:
    #     - Check IOU
    #     - If IOU > threshold, perform feature matching
    #     - Remove duplicates based on feature similarity and box size
    #     """
    #     # If det_close is True, immediately return without doing anything
        
    #     # First, remove boxes with width less than 90
    #     frame_width = frame.shape[1]
    
    #     # Calculate the 10% threshold from the left edge
    #     left_edge_threshold = frame_width * 0.1
    #     if self.det_close:
    #         return
    #     # Filter persons with updated conditions
    #     self.persons_inside = [
    #         person for person in self.persons_inside
    #         if (person.bbox is not None and 
    #             (person.bbox[2] >= 90 or  # Keep bbox if width >= 90
    #             person.bbox[0] <= left_edge_threshold))  # Or if near left edge
    #     ]
        

    #     if self.det_intersect == True:
    #         return

    #     to_remove = []
    #     # Compare each pair of remaining persons
    #     for i in range(len(self.persons_inside)):
    #         for j in range(i + 1, len(self.persons_inside)):
    #             person1 = self.persons_inside[i]
    #             person2 = self.persons_inside[j]

    #             # Only proceed if both persons are in frame
    #             if not (person1.inframe and person2.inframe):
    #                 continue

    #             # Calculate IOU
    #             iou = self.calculate_iou(person1.bbox, person2.bbox)
                
    #             # If IOU is greater than threshold (25%)
    #             if iou > 0.25:
    #                 # Calculate feature centroids
    #                 _, _, similarity1 = self.calculate_feature_centroid(person1.image, person2.image)
                    
    #                 # Check if feature centroids are similar
    #                 if similarity1 > 35:
    #                     # Calculate box dimensions
    #                     _, _, w1, h1 = person1.bbox
    #                     _,_, w2, h2 = person2.bbox

    #                     # Determine which person to remove based on box size
    #                     area1 = w1 * h1
    #                     area2 = w2 * h2
                        
    #                     if area1 > area2 and j not in to_remove:
    #                         to_remove.append(j)
    #                     elif area2 > area1 and i not in to_remove:
    #                         to_remove.append(i)

    #     # Remove marked persons in reverse order to maintain correct indices
    #     for index in sorted(to_remove, reverse=True):
    #         del self.persons_inside[index]
    # def remove_duplicates_iou(self,frame):
    #     """
    #     Remove duplicate detections based on multiple criteria:
    #     1. First, remove all boxes with width less than 90
    #     2. Then, for remaining boxes:
    #     - Check IOU
    #     - If IOU > threshold, perform feature matching
    #     - Remove duplicates based on feature similarity and box size
    #     """
    #     # If det_close is True, immediately return without doing anything
        
    #     # First, remove boxes with width less than 90
    #     frame_width = frame.shape[1]
    
    #     # Calculate the 10% threshold from the left edge
    #     left_edge_threshold = frame_width * 0.1
    #     # if self.det_close:
    #     #     return
    #     # Filter persons with updated conditions
    #     self.persons_inside = [
    #         person for person in self.persons_inside
    #         if (person.bbox is not None and 
    #             (person.bbox[2] >= 90 ))  # Or if near left edge
    #     ]
        

    #     # if self.det_intersect == True:
    #     #     return

    #     to_remove = []
    #     # Compare each pair of remaining persons
    #     for i in range(len(self.persons_inside)):
    #         for j in range(i + 1, len(self.persons_inside)):
    #             person1 = self.persons_inside[i]
    #             person2 = self.persons_inside[j]

    #             # Only proceed if both persons are in frame
    #             if not (person1.inframe and person2.inframe):
    #                 continue

    #             # Calculate IOU
    #             iou = self.calculate_iou(person1.bbox, person2.bbox)
                
    #             # If IOU is greater than threshold (25%)
    #             if iou > 0.25:
    #                 # Calculate feature centroids
    #                 _, _, similarity1 = self.calculate_feature_centroid(person1.image, person2.image)
                    
    #                 # Check if feature centroids are similar
    #                 if similarity1 > 35:
    #                     # Calculate box dimensions
    #                     _, _, w1, h1 = person1.bbox
    #                     _,_, w2, h2 = person2.bbox

    #                     # Determine which person to remove based on box size
    #                     area1 = w1 * h1
    #                     area2 = w2 * h2
                        
    #                     if area1 > area2 and j not in to_remove:
    #                         to_remove.append(j)
    #                     elif area2 > area1 and i not in to_remove:
    #                         to_remove.append(i)

    #     # Remove marked persons in reverse order to maintain correct indices
    #     for index in sorted(to_remove, reverse=True):
    #         del self.persons_inside[index]
    #     to_remove = []
    #     for i in range(len(self.persons_leaving)):
    #         for j in range(i + 1, len(self.persons_leaving)):
    #             person1 = self.persons_leaving[i]
    #             person2 = self.persons_leaving[j]

    #             # Only proceed if both persons are in frame
    #             # if not (person1.inframe and person2.inframe):
    #             #     continue

    #             # Calculate IOU
    #             iou = self.calculate_iou(person1.bbox, person2.bbox)
                
    #             # If IOU is greater than threshold (25%)
    #             if iou > 0.25:
    #                 # Calculate feature centroids
    #                 _, _, similarity1 = self.calculate_feature_centroid(person1.image, person2.image)
                    
    #                 # Check if feature centroids are similar
    #                 if similarity1 > 35:
    #                     # Calculate box dimensions
    #                     _, _, w1, h1 = person1.bbox
    #                     _,_, w2, h2 = person2.bbox

    #                     # Determine which person to remove based on box size
    #                     area1 = w1 * h1
    #                     area2 = w2 * h2
                        
    #                     if area1 > area2 and j not in to_remove:
    #                         to_remove.append(j)
    #                     elif area2 > area1 and i not in to_remove:
    #                         to_remove.append(i)

    #     # Remove marked persons in reverse order to maintain correct indices
    #     for index in sorted(to_remove, reverse=True):
    #         del self.persons_leaving[index]
    def remove_duplicates_iou(self,frame):
        """
        Remove duplicate detections and persons who haven't been detected for too long.
        Criteria:
        1. Remove persons undetected for more than 10 frames
        2. Remove all boxes with width less than 90
        3. For remaining boxes:
            - Check IOU
            - If IOU > threshold, perform feature matching
            - Remove duplicates based on feature similarity and box size
        """
        frame_width = frame.shape[1]
        left_edge_threshold = frame_width * 0.1

        # First remove persons who haven't been detected for more than 10 frames
        # self.persons_inside = [
        #     person for person in self.persons_inside
        #     if not (hasattr(person, 'undetected') and person.undetected > 10)
        # ]
        
        # self.persons_leaving = [
        #     person for person in self.persons_leaving
        #     if not (hasattr(person, 'undetected') and person.undetected > 10)
        # ]

        # Then filter persons based on bbox width
        self.persons_inside = [
            person for person in self.persons_inside
            if (person.bbox is not None and 
                (person.bbox[2] >= 90))
        ]
        self.persons_leaving = [
            person for person in self.persons_leaving
            if (person.bbox is not None and 
                (person.bbox[2] >= 90))
        ]

        to_remove = []
        # Compare each pair of remaining persons inside
        for i in range(len(self.persons_inside)):
            for j in range(i + 1, len(self.persons_inside)):
                person1 = self.persons_inside[i]
                person2 = self.persons_inside[j]

                # Only proceed if both persons are in frame
                if not (person1.inframe and person2.inframe):
                    continue

                # Calculate IOU
                iou = self.calculate_iou(person1.bbox, person2.bbox)
                
                # If IOU is greater than threshold (25%)
                if iou > 0.25:
                    # Calculate feature centroids
                    _, _, similarity1 = self.calculate_feature_centroid(person1.image, person2.image)
                    
                    # Check if feature centroids are similar
                    if similarity1 > 35:
                        # Calculate box dimensions
                        _, _, w1, h1 = person1.bbox
                        _,_, w2, h2 = person2.bbox

                        # Determine which person to remove based on box size
                        area1 = w1 * h1
                        area2 = w2 * h2
                        
                        if area1 > area2 and j not in to_remove:
                            to_remove.append(j)
                        elif area2 > area1 and i not in to_remove:
                            to_remove.append(i)

        # Remove marked persons in reverse order to maintain correct indices
        for index in sorted(to_remove, reverse=True):
            del self.persons_inside[index]

        # Reset to_remove for persons_leaving
        to_remove = []
        
        # Compare each pair of remaining persons leaving
        for i in range(len(self.persons_leaving)):
            for j in range(i + 1, len(self.persons_leaving)):
                person1 = self.persons_leaving[i]
                person2 = self.persons_leaving[j]

                # Calculate IOU
                iou = self.calculate_iou(person1.bbox, person2.bbox)
                
                # If IOU is greater than threshold (25%)
                if iou > 0.25:
                    # Calculate feature centroids
                    _, _, similarity1 = self.calculate_feature_centroid(person1.image, person2.image)
                    
                    # Check if feature centroids are similar
                    if similarity1 > 35:
                        # Calculate box dimensions
                        _, _, w1, h1 = person1.bbox
                        _,_, w2, h2 = person2.bbox

                        # Determine which person to remove based on box size
                        area1 = w1 * h1
                        area2 = w2 * h2
                        
                        if area1 > area2 and j not in to_remove:
                            to_remove.append(j)
                        elif area2 > area1 and i not in to_remove:
                            to_remove.append(i)

        # Remove marked persons in reverse order to maintain correct indices
        for index in sorted(to_remove, reverse=True):
            del self.persons_leaving[index]
    # def remove_duplicates_iou(self):   #### make it based on only union area  aklso onlly do for person_inframe-= e
    #     """
    #     Remove duplicate detections based on IOU threshold.
    #     Keeps the person with the larger bounding box when duplicates are found.
    #     """
    #     if self.det_close:
    #         return
    #     to_remove = []
        
    #     # Compare each pair of persons
    #     for i in range(len(self.persons_inside)):
    #         for j in range(i + 1, len(self.persons_inside)):
    #             person1 = self.persons_inside[i]
    #             person2 = self.persons_inside[j]
                
    #             # Skip if either person doesn't have a bbox
    #             if person1.bbox is None or person2.bbox is None:
    #                 continue
                
    #             # Calculate IOU
    #             iou = self.calculate_iou(person1.bbox, person2.bbox)
                
    #             # If IOU is greater than threshold (25%)
    #             if iou > 0.25:
    #                 # Calculate box areas
    #                 _, _, w1, h1 = person1.bbox
    #                 _, _, w2, h2 = person2.bbox
    #                 area1 = w1 * h1
    #                 area2 = w2 * h2
                    
    #                 # Mark smaller box for removal
    #                 if area1 > area2 and j not in to_remove:
    #                     to_remove.append(j)
    #                 elif area2 > area1 and i not in to_remove:
    #                     to_remove.append(i)
    #                 # If equal size, remove the second one
    #                 elif i not in to_remove:
    #                     to_remove.append(i)
        
    # # Remove marked persons in reverse order to maintain correct indices
    #     for index in sorted(to_remove, reverse=True):
    #         del self.persons_inside[index]
   
    

    def display_person_images(self):
        """
        Display and save the images of each person in the persons_inside list using cv2.
        Each person's image will be shown in a separate window and saved as person_id.jpg
        """
        # Create a directory for saving images if it doesn't exist
        save_dir = "person_images"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, person in enumerate(self.persons_inside):
            if person.image is not None:
                # Create a window name with person's index and ID
                window_name = f"Person {i} (ID: {person.person_id})"
                
                # Show the image
                #cv2.imshow(window_name, person.image)
                window = cv2.getWindowImageRect(window_name)
                
                # Set the window property to overlay text
                cv2.displayOverlay(window_name, f"In Frame: {person.inframe}", 0)
                
                # Save the image with person_id as filename in the specified directory
                save_path = os.path.join(save_dir, f"person_{person.person_id}.jpg")
                cv2.imwrite(save_path, person.image)
                print(f"Saved image for Person {person.person_id} as {save_path}")
                
                # Wait for a key press
                cv2.waitKey(1)
        # for i, person in enumerate(self.persons_leaving):
        #     if person.image is not None:
        #         # Create a window name with person's index and ID
        #         window_name = f"Person {i} (ID: {person.person_id})"
                
        #         # Show the image
        #         cv2.imshow(window_name, person.image)
        #         window = cv2.getWindowImageRect(window_name)
            
        #         # Set the window property to overlay text
        #         # This adds text to the window itself, not the image
        #         cv2.displayOverlay(window_name, f"In Frame: {person.inframe}", 0)
        #         # Wait for a key press (0 means wait indefinitely)
        #         # You may want to adjust this if you want the images to show for a specific duration
        #         cv2.waitKey(1000)
        #         cv2.destroyWindow(window_name)
               
        # for i, person in enumerate(self.persons_inside_room):
        #     if person.image is not None:
        #         # Create a window name with person's index and ID
        #         window_name = f"Person {i} (ID: {person.person_id})"
                
        #         # Show the image
        #         cv2.imshow(window_name, person.image)
        #         window = cv2.getWindowImageRect(window_name)
            
        #         # Set the window property to overlay text
        #         # This adds text to the window itself, not the image
        #         cv2.displayOverlay(window_name, f"In Frame: {person.inframe}", 0)
        #         # Wait for a key press (0 means wait indefinitely)
        #         # You may want to adjust this if you want the images to show for a specific duration
        #         cv2.waitKey(1)
        # Close all windows after displaying
        
   
    def add_people(self, results, frame):
        """
        Add people from YOLO detection results to the person list.
        Args:
        results: YOLO detection results
        frame (numpy.ndarray): Current frame
        Returns:
        None: Creates global variables person1, person2, etc.
        """
        # Reset the person counter at the start
        self.person_counter = 1
        self.check_person_proximity()
        if self.det_intersect  == False :

            for i, result in enumerate(results):
                # Get bounding boxes
                boxes = result.boxes
                
                # Filter for person class (class index 0 in COCO dataset)
                for j, box in enumerate(boxes):
                    # Check if the detected class is a person
                    if int(box.cls[0]) == 0:  # 0 is the index for 'person' in COCO dataset
                        # Get bounding box coordinates
                        x, y, w, h = box.xywh[0].tolist()
                        
                        # Convert to integers
                        bbox = (int(x-w/2), int(y-h/2), int(w), int(h))
                        
                        # Only create person object if width is greater than 90
                        if int(h) > 400:
                                # Calculate center
                            center = self.get_bounding_box_center(bbox)
                            
                            # Dynamically create variable names like person1
                            person_var_name = f'person{self.person_counter}'
                            
                            # Extract the bounding box image from the frame
                            x, y, width, height = bbox
                            image = frame[y:y+height, x:x+width]
                            
                            # Create Person object and assign to a global variable
                            globals()[person_var_name] = Person(
                                bbox=bbox,
                                first_center=center,
                                person_id=self.person_counter,
                                inframe=True,
                                image=image
                            )
                            
                            # Check entry for this person
                            self.check_entry(globals()[person_var_name])
                            
                            # Increment person counter
                            self.person_counter += 1

    # def check_person_proximity(self, threshold=50):
    #     """
    #     Check proximity and intersection of person bounding boxes in persons_inside list
    #     and update det_close and det_intersect.
    #     """
    #     if len(self.persons_inside) < 2:
    #         self.det_close = False
    #         self.det_intersect = False
    #         return
            
    #     self.det_close = False
    #     self.det_intersect = False
      
    #     for i, person1 in enumerate(self.persons_inside):
    #         for j, person2 in enumerate(self.persons_inside):
    #             if i != j and person1.bbox is not None and person2.bbox is not None:
    #                 x1, y1, w1, h1 = person1.bbox
    #                 x2, y2, w2, h2 = person2.bbox
                    
    #                 # Calculate edgesas
    #                 left = max(x1, x2)
    #                 right = min(x1 + w1, x2 + w2)
    #                 top = max(y1, y2)
    #                 bottom = min(y1 + h1, y2 + h2)
    #                 # iou=self.calculate_iou(person1.bbox,person2.bbox)
    #                 # self.iou=iou
    #                # print(iou, "shhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    #                 # Calculate intersection area
    #                 if left < right and top < bottom:
    #                     intersection_area = (right - left) * (bottom - top)
    #                     if intersection_area > 0:
    #                         self.det_intersect = True
    #                         self.det_close = True
    #                         return
    #                 # if iou > 0 :
    #                 #     self.det_intersect = True   
    #                 # Check proximity if no intersection
    #                 dx = min(abs(x1 - (x2 + w2)), abs(x2 - (x1 + w1)))
    #                 dy = min(abs(y1 - (y2 + h2)), abs(y2 - (y1 + h1)))
    #                 min_distance = min(dx, dy)
                    
    #                 if min_distance <= threshold:
    #                     self.det_close = True
    #                     if self.det_intersect:
    #                         return

    # def check_person_proximity(self, threshold=50):### in fram e
    #     """
    #     Check proximity and intersection of person bounding boxes in persons_inside list
    #     and update det_close and det_intersect. Only compares persons where inframe is True.
    #     """
    #     # Reset detection flags
    #     self.det_close = False
    #     self.det_intersect = False
        
    #     # Check if we have at least 2 persons to compare
    #     if len(self.persons_inside) < 2:
    #         return
            
    #     for i, person1 in enumerate(self.persons_inside):
    #         for j, person2 in enumerate(self.persons_inside):
    #             # Skip if same person or if either person is not in frame
    #             if (i == j or 
    #                 not person1.inframe or 
    #                 not person2.inframe or
    #                 person1.bbox is None or 
    #                 person2.bbox is None):
    #                 continue
                    
    #             # Extract bounding box coordinates
    #             x1, y1, w1, h1 = person1.bbox
    #             x2, y2, w2, h2 = person2.bbox
                
    #             # Calculate intersection edges
    #             left = max(x1, x2)
    #             right = min(x1 + w1, x2 + w2)
    #             top = max(y1, y2)
    #             bottom = min(y1 + h1, y2 + h2)
                
    #             # Check for intersection
    #             if left < right and top < bottom:
    #                 intersection_area = (right - left) * (bottom - top)
    #                 if intersection_area > 0:
    #                     self.det_intersect = True
    #                     self.det_close = True
    #                     return
                
    #             # Check proximity if no intersection
    #             dx = min(abs(x1 - (x2 + w2)), abs(x2 - (x1 + w1)))
    #             dy = min(abs(y1 - (y2 + h2)), abs(y2 - (y1 + h1)))
    #             min_distance = min(dx, dy)
                
    #             if min_distance <= threshold:
    #                 self.det_close = True
    #                 if self.det_intersect:
    #                     return
    def check_person_proximity(self, threshold=50):  ### checking across lists 
        """
        Check proximity and intersection of person bounding boxes in both persons_inside
        and persons_leaving lists. Updates det_close and det_intersect flags.
        Only compares persons where inframe is True.
        """
        # Reset detection flags
        self.det_close = False
        self.det_intersect = False
        
        # First check within persons_inside
        for i, person1 in enumerate(self.persons_inside):
            for j, person2 in enumerate(self.persons_inside):
                if (i == j or
                    not person1.inframe or
                    not person2.inframe or
                    person1.bbox is None or
                    person2.bbox is None):
                    continue
                    
                if self._check_pair_proximity(person1, person2, threshold):
                    return

        # Then check within persons_leaving
        for i, person1 in enumerate(self.persons_leaving):
            for j, person2 in enumerate(self.persons_leaving):
                if (i == j or
                    not person1.inframe or
                    not person2.inframe or
                    person1.bbox is None or
                    person2.bbox is None):
                    continue
                    
                if self._check_pair_proximity(person1, person2, threshold):
                    return

        # Finally check between the two lists
        for person1 in self.persons_inside:
            for person2 in self.persons_leaving:
                if (not person1.inframe or
                    not person2.inframe or
                    person1.bbox is None or
                    person2.bbox is None):
                    continue
                    
                if self._check_pair_proximity(person1, person2, threshold):
                    return

    def _check_pair_proximity(self, person1, person2, threshold):
        """
        Helper method to check proximity between two persons.
        Returns True if detection found (close or intersecting).
        """
        x1, y1, w1, h1 = person1.bbox
        x2, y2, w2, h2 = person2.bbox
        
        # Calculate intersection edges
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        # Check for intersection
        if left < right and top < bottom:
            intersection_area = (right - left) * (bottom - top)
            if intersection_area > 0:
                self.det_intersect = True
                self.det_close = True
                return True
                
        # Check proximity if no intersection
        dx = min(abs(x1 - (x2 + w2)), abs(x2 - (x1 + w1)))
        dy = min(abs(y1 - (y2 + h2)), abs(y2 - (y1 + h1)))
        min_distance = min(dx, dy)
        
        if min_distance <= threshold:
            self.det_close = True
            return True
            
        return False
    # def check_person_proximity(self, threshold=50):  #####intresect if iou > 30 
    #     """
    #     Check proximity and intersection of person bounding boxes in persons_inside list
    #     and update det_close and det_intersect.
    #     Args:
    #         threshold: Maximum distance between bbox edges to be considered close (default 50)
    #     """
    #     if len(self.persons_inside) < 2:
    #         self.det_close = False
    #         self.det_intersect = False
    #         return

    #     self.det_close = False
    #     self.det_intersect = False

    #     for i, person1 in enumerate(self.persons_inside):
    #         for j, person2 in enumerate(self.persons_inside):
    #             if i != j and person1.bbox is not None and person2.bbox is not None:
    #                 x1, y1, w1, h1 = person1.bbox
    #                 x2, y2, w2, h2 = person2.bbox

    #                 # Check proximity
    #                 left_to_right = abs((x1) - (x2 + w2))
    #                 right_to_left = abs((x1 + w1) - x2)
    #                 top_to_bottom = abs(y1 - (y2 + h2))
    #                 bottom_to_top = abs((y1 + h1) - y2)

    #                 if (left_to_right <= threshold or
    #                     right_to_left <= threshold or
    #                     top_to_bottom <= threshold or
    #                     bottom_to_top <= threshold):
    #                     self.det_close = True

    #                 # Calculate intersection
    #                 x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    #                 y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    
    #                 if x_overlap > 0 and y_overlap > 0:
    #                     # Calculate intersection area
    #                     intersection_area = x_overlap * y_overlap
                        
    #                     # Calculate box areas
    #                     area1 = w1 * h1
    #                     area2 = w2 * h2
                        
    #                     # Calculate IoU
    #                     union_area = area1 + area2 - intersection_area
    #                     iou = (intersection_area / union_area) * 100
                        
    #                     # Update det_intersect only if IoU > 30%
    #                     if iou > 15:
    #                         self.det_intersect = True
                            
    #                 if self.det_close and self.det_intersect:
    #                     return
    def check_boxes(self,bbox1, bbox2,threshold=50):
        """Helper function to check proximity and intersection of two boxes"""
        # Unpack bounding box coordinates
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate distances between edges
        left_to_right = abs((x1) - (x2 + w2))
        right_to_left = abs((x1 + w1) - x2)
        top_to_bottom = abs(y1 - (y2 + h2))
        bottom_to_top = abs((y1 + h1) - y2)
        
        # Check proximity
        if (left_to_right <= threshold or
            right_to_left <= threshold or
            top_to_bottom <= threshold or
            bottom_to_top <= threshold):
            return True, False  # Close but not necessarily intersecting
            
        # Check intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        # Return intersection status
        return False, (x_overlap > 0 and y_overlap > 0)
    # def check_person_proximity(self, threshold=50):
    #     """
    #     Check proximity and intersection of person bounding boxes in both persons_inside
    #     and persons_inside_room lists and update det_close and det_intersect.
        
    #     Args:
    #         threshold: Maximum distance between bbox edges to be considered close (default 50)
    #     """
    #     # Combine total number of persons to check
    #     total_persons = len(self.persons_inside) + len(self.persons_inside_room)
        
    #     # If less than 2 total persons, no need to check
    #     if total_persons < 2:
    #         self.det_close = False
    #         self.det_intersect = False
    #         return
        
    #     # Reset flags
    #     self.det_close = False
    #     self.det_intersect = False
        
        
        
    #     # Check between persons_inside
    #     for i, person1 in enumerate(self.persons_inside):
    #         # First check within persons_inside
    #         for j, person2 in enumerate(self.persons_inside):
    #             if i != j and person1.bbox is not None and person2.bbox is not None:
    #                 is_close, is_intersect = self.check_boxes(person1.bbox, person2.bbox)
    #                 if is_close:
    #                     self.det_close = True
    #                 if is_intersect:
    #                     self.det_intersect = True
    #                 if self.det_close and self.det_intersect:
    #                     return
                        
    #         # Then check against persons_inside_room
    #         for person2 in self.persons_inside_room:
    #             if person1.bbox is not None and person2.bbox is not None:
    #                 is_close, is_intersect = self.check_boxes(person1.bbox, person2.bbox)
    #                 if is_close:
    #                     self.det_close = True
    #                 if is_intersect:
    #                     self.det_intersect = True
    #                 if self.det_close and self.det_intersect:
    #                     return
        
    #     # Check between persons_inside_room if needed
    #     for i, person1 in enumerate(self.persons_inside_room):
    #         for j, person2 in enumerate(self.persons_inside_room):
    #             if i != j and person1.bbox is not None and person2.bbox is not None:
    #                 is_close, is_intersect = self.check_boxes(person1.bbox, person2.bbox)
    #                 if is_close:
    #                     self.det_close = True
    #                 if is_intersect:
    #                     self.det_intersect = True
    #                 if self.det_close and self.det_intersect:
    #                     return 
    # def preprocess(self, img):
    #     """
    #     Preprocess the image: convert to grayscale (if needed) and resize to (640, 480).
    #     Returns both the resized image and the scaling factors.
    #     """
    #     original_shape = img.shape[:2]  # Original height, width
    #     if len(img.shape) == 3:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img_resized = cv2.resize(img, (640, 480))
        
    #     scale_x = original_shape[1] / 640  # Scaling factor for width
    #     scale_y = original_shape[0] / 480  # Scaling factor for height
        
    #     return img_resized, torch.from_numpy(img_resized).float().to(self.device)[None, None] / 255.0, (scale_x, scale_y)

    def match_and_get_centroids(self, image0, image1):
        """
        Match two images and return centroids and matching info in original resolution.
        Args:
            image0: First input image (numpy array)
            image1: Second input image (numpy array)
        Returns:
            dict with centroids, matching points, and match count
        """
        # Preprocess images and get scaling factors
        img0, inp0, scale0 = self.preprocess(image0)
        img1, inp1, scale1 = self.preprocess(image1)
        
        with torch.no_grad():
            pred = self.matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        # Scale keypoints back to the original image resolution
        mkpts0[:, 0] *= scale0[0]  # Scale x-coordinates for image0
        mkpts0[:, 1] *= scale0[1]  # Scale y-coordinates for image0
        
        mkpts1[:, 0] *= scale1[0]  # Scale x-coordinates for image1
        mkpts1[:, 1] *= scale1[1]  # Scale y-coordinates for image1
        
        centroid0 = np.mean(mkpts0, axis=0) if len(mkpts0) > 0 else None
        centroid1 = np.mean(mkpts1, axis=0) if len(mkpts1) > 0 else None
        # plt.figure(figsize=(15, 7))
    
        # plt.subplot(121)
        # plt.imshow(image0, cmap='gray')
        # plt.scatter(mkpts0[:, 0], mkpts0[:, 1], c='r', s=5)
        # if centroid0 is not None:
        #     plt.scatter(centroid0[0], centroid0[1], c='g', s=100, marker='*', label='Centroid')
        # plt.title('Reference Image')
        # plt.legend()
        
        # plt.subplot(122)
        # plt.imshow(image1, cmap='gray')
        # plt.scatter(mkpts1[:, 0], mkpts1[:, 1], c='r', s=5)
        # if centroid1 is not None:
        #     plt.scatter(centroid1[0], centroid1[1], c='g', s=100, marker='*', label='Centroid')
        
        # for pt0, pt1 in zip(mkpts0, mkpts1):
        #     plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'b-', linewidth=0.5, alpha=0.5)
        
        # plt.title('Query Image with Matches')
        # plt.legend()
        # plt.show()
        return {
            'centroid0': centroid0,
            'centroid1': centroid1,
            'matches0': mkpts0,
            'matches1': mkpts1,
            'num_matches': len(mkpts0)
        }
    def check_ppl_inside(self):
        """
        Return the count of people inside the list.
        
        Returns:
            int: Number of people inside
        """
        # print(len(self.persons_inside), "insiode",len (self.persons_inside_room) , "inside room ")
        count=len(self.persons_inside) 
        return count
    def check_union_area(self, box1, box2, min_union_ratio=0.8):
        """
        Check if the union area is at least a specified ratio of the smaller bounding box.
        Args:
            box1: tuple (x1, y1, w1, h1)
            box2: tuple (x2, y2, w2, h2)
            min_union_ratio: minimum ratio of union area to smaller bbox area (default 0.8)
        Returns:
            bool: True if union area meets the criteria, False otherwise
        """
        # Calculate box areas
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1, w2, h2 = box2
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Calculate intersection coordinates
        x1_2 = x1_1 + w1
        y1_2 = y1_1 + h1
        x2_2 = x2_1 + w2
        y2_2 = y2_1 + h2

        # Calculate intersection coordinates
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)

        # If no intersection, return False
        if x_right < x_left or y_bottom < y_top:
            return False

        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Determine the smaller box area
        smaller_area = min(box1_area, box2_area)

        # Check if union area meets the minimum ratio of the smaller box
        return (union_area / smaller_area) >= min_union_ratio
    # def track_all_persons(self, frame):
    #     """
    #     Track all persons in the persons_inside list.
    #     Updates the bounding box for each person based on their tracker output.
    #     Args:
    #     frame (numpy.ndarray): Current frame for tracking
    #     """
    #     frame2 = frame.copy()
    #     frame_width = frame.shape[1]
    #     x_threshold = int(frame_width * 0.1)

    #     for person in self.persons_inside:
    #         # Unpack previous bbox
    #         x, y, w, h = person.bbox

    #         # Crop the portion of the frame corresponding to the previous bbox
    #         # Calculate feature centroid
    #         centroid_x, centroid_y, confidence = self.calculate_feature_centroid(person.image, frame2)

    #         # Check if confidence is more than 25
    #         if confidence > 35:
    #             # Use centroid to create a new tracked bbox
    #             tracked_bbox = (
    #                 int(centroid_x - w/2),
    #                 int(centroid_y - h/2),
    #                 w,
    #                 h
    #             )
    #             person.bbox = tracked_bbox

    #             # Check if the x-coordinate of the bbox centroid is less than 10% of frame width
    #             bbox_left_x = tracked_bbox[0]
    #             if bbox_left_x < x_threshold:
    #                 person.inframe = False
                
    #             # Draw tracker around the object on the original frame
    #             cv2.rectangle(
    #                 frame, 
    #                 (tracked_bbox[0], tracked_bbox[1]), 
    #                 (tracked_bbox[0] + tracked_bbox[2], tracked_bbox[1] + tracked_bbox[3]), 
    #                 (0, 0, 255),  # red
    #                 2  # Line thickness
    #             )
    # def track_all_persons(self, frame):   ##########with left thresholf 
    #     """
    #     Track all persons in the persons_inside list.
    #     Updates the bounding box for each person based on their tracker output.
    #     Moves persons leaving the frame to persons_inside_room list.
        
    #     Args:
    #     frame (numpy.ndarray): Current frame for tracking
    #     """
    #     # Initialize persons_inside_room if it doesn't exist
    #     if not hasattr(self, 'persons_inside_room'):
    #         self.persons_inside_room = []
        
        
        
    #     frame2 = frame.copy()
    #     frame_width = frame.shape[1]
    #     x_threshold = int(frame_width * 0.1)
        
    #     # Create a list to track indices of persons to remove
    #     persons_to_remove = []
        
    #     for i, person in enumerate(self.persons_inside):
    #         if person.inframe:
    #             # Unpack previous bbox
    #             x, y, w, h = person.bbox
                
    #             # Calculate feature centroid
    #             centroid_x, centroid_y, confidence = self.calculate_feature_centroid(person.image, frame2)
                
    #             # Check if confidence is more than 25
    #             if confidence > 35:
    #                 # Use centroid to create a new tracked bbox
    #                 tracked_bbox = (
    #                     int(centroid_x - w/2),
    #                     int(centroid_y - h/2),
    #                     w,
    #                     h
    #                 )
    #                 person.bbox = tracked_bbox
                    
    #                 # Check if the x-coordinate of the bbox centroid is less than 10% of frame width
    #                 bbox_left_x = tracked_bbox[0]
    #                 if bbox_left_x < x_threshold:
    #                     # Mark this person for removal from persons_inside
    #                     # persons_to_remove.append(i)
    #                     # # Add to persons_inside_room
    #                     # self.persons_inside_room.append(person)

    #                     person.inframe = False
                    
    #                 # Draw tracker around the object on the original frame
    #                 cv2.rectangle(
    #                     frame,
    #                     (tracked_bbox[0], tracked_bbox[1]),
    #                     (tracked_bbox[0] + tracked_bbox[2], tracked_bbox[1] + tracked_bbox[3]),
    #                     (255, 0, 255),  # red
    #                     2  # Line thickness
    #                 )
        
    #     # Remove persons from persons_inside in reverse order to maintain correct indices
    #     for index in sorted(persons_to_remove, reverse=True):
    #         del self.persons_inside[index]
    # def track_all_persons(self, frame):
    #     """
    #     Track all persons in the persons_inside list.
    #     Updates the bounding box for each person based on their tracker output.
    #     Moves persons leaving the frame to persons_inside_room list.
        
    #     Args:
    #     frame (numpy.ndarray): Current frame for tracking
    #     """
    #     # Initialize persons_inside_room if it doesn't exist
    #     if not hasattr(self, 'persons_inside_room'):
    #         self.persons_inside_room = []
        
        
        
    #     frame2 = frame.copy()
    #     frame_width = frame.shape[1]
    #     x_threshold = int(frame_width * 0.1)
        
    #     # Create a list to track indices of persons to remove
    #     persons_to_remove = []
        
    #     for i, person in enumerate(self.persons_inside):
    #         # Unpack previous bbox
    #         x, y, w, h = person.bbox
            
    #         # Calculate feature centroid
    #         centroid_x, centroid_y, confidence = self.calculate_feature_centroid(person.image, frame2)
            
    #         # Check if confidence is more than 25
    #         if confidence > 65:  #25
    #             # Use centroid to create a new tracked bbox
    #             tracked_bbox = (
    #                 int(centroid_x - w/2),
    #                 int(centroid_y - h/2),
    #                 w,
    #                 h
    #             )
    #             person.bbox = tracked_bbox
                
    #             # Check if the x-coordinate of the bbox centroid is less than 10% of frame width
               
                
    #             # Draw tracker around the object on the original frame
    #             cv2.rectangle(
    #                 frame,
    #                 (tracked_bbox[0], tracked_bbox[1]),
    #                 (tracked_bbox[0] + tracked_bbox[2], tracked_bbox[1] + tracked_bbox[3]),
    #                 (255, 0, 255),  # red
    #                 2  # Line thickness
    #             )
        
    #     # Remove persons from persons_inside in reverse order to maintain correct indices
    #     for index in sorted(persons_to_remove, reverse=True):
    #         del self.persons_inside[index]
    def checks(self):
        """
        Remove persons from tracking lists based on tracking conditions:
        - For persons_leaving: Remove if undetected > 3 frames and crossed door threshold
        - For persons_inside: Remove if undetected > 10 frames or crossed door threshold
        """
        door_threshold = self.door[0] + self.door[2]  # Right edge of door
        
        # Process persons_leaving list
        persons_leaving_to_remove = []
        for i, person in enumerate(self.persons_leaving):
            print(person.person_id, "lea\ving gggggggggg")
            
            if person.undetected > 25:
                person.inframe = False
                if person.bbox[0] + (person.bbox[2]/2) > door_threshold:
                    persons_leaving_to_remove.append(i)
                else:
                    persons_leaving_to_remove.append(i)
                    # Remove one person from persons_inside if available
                    if len(self.persons_inside) > 0:
                        self.persons_inside.pop()
        
        # Remove marked persons from persons_leaving list
        for index in sorted(persons_leaving_to_remove, reverse=True):
            del self.persons_leaving[index]
        
        # Process persons_inside list
        persons_inside_to_remove = []
        for i, person in enumerate(self.persons_inside):
            # Check if person crossed door threshold
            print(person.person_id, "inside eeeeeeeeeeeee")
            if person.bbox[0] + person.bbox[2]/2 > door_threshold:
                person.inframe = False
                
            # Check if person was undetected for too long
            elif person.undetected > 25:
                person.inframe = False
                
                
                persons_inside_to_remove.append(i)
        
        # Remove marked persons from persons_inside list
        for index in sorted(persons_inside_to_remove, reverse=True):
            del self.persons_inside[index]
    def track_all_persons(self, frame):
        """
        Track all persons in the persons_inside list.
        Updates the bounding box for each person based on their tracker output.
        Makes person.inframe False if they move beyond door threshold.
        Removes person from persons_inside if they move left of door.
        
        Args:
            frame (numpy.ndarray): Current frame for tracking
        """
        if not hasattr(self, 'persons_inside_room'):
            self.persons_inside_room = []
            
        frame2 = frame.copy()
        frame_width = frame.shape[1]
        door_threshold = self.door[0] + self.door[2]  # Right edge of door
        door_left_edge = self.door[0]  # Left edge of door
        persons_to_remove = []

        # Track persons inside
        for i, person in enumerate(self.persons_inside):
            if person.inframe:
                x, y, w, h = person.bbox
                centroid_x, centroid_y, confidence = self.calculate_feature_centroid(person.image, frame2)
                if confidence > 35:
                    tracked_bbox = (
                        int(centroid_x - w/2),
                        int(centroid_y - h/2),
                        w,
                        h
                    )
                    person.bbox = tracked_bbox
                    person.tracking = True
                    
                    # Check if person has moved beyond door threshold
                    if tracked_bbox[0] + tracked_bbox[2]/2 > door_threshold:
                        person.inframe = False
                    if person.undetected > 10 :
                        person.inframe = False
                        
                    # Draw tracking visualization if still in frame
                    cv2.rectangle(
                        frame,
                        (tracked_bbox[0], tracked_bbox[1]),
                        (tracked_bbox[0] + tracked_bbox[2], tracked_bbox[1] + tracked_bbox[3]),
                        (255, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        str(person.undetected),
                        (tracked_bbox[0], tracked_bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # Track persons leaving
        persons_leaving_to_remove = []
        for i, person in enumerate(self.persons_leaving):
            if person.inframe:
                x, y, w, h = person.bbox
                centroid_x, centroid_y, confidence = self.calculate_feature_centroid(person.image, frame2)
                if confidence > 35:
                    tracked_bbox = (
                        int(centroid_x - w/2),
                        int(centroid_y - h/2),
                        w,
                        h
                    )
                    person.bbox = tracked_bbox
                    person.tracking = True
                    
                    # Check if person has moved beyond door threshold
                    if tracked_bbox[0] + (tracked_bbox[2]/2) < door_left_edge:
                        person.inframe = False

                        persons_leaving_to_remove.append(i)
                        # Remove one person from persons_inside
                        if len(self.persons_inside) > 0:
                            self.persons_inside.pop()
                    if person.undetected > 3 :
                        person.inframe = False
                        if person.bbox[0] + (person.bbox[2]/2) > door_threshold:
                            persons_leaving_to_remove.append(i)
                        else:
                            persons_leaving_to_remove.append(i)
                        # Remove one person from persons_inside
                            if len(self.persons_inside) > 0:
                                self.persons_inside.pop()
                    # Draw tracking visualization if still in frame
                    cv2.rectangle(
                        frame,
                        (tracked_bbox[0], tracked_bbox[1]),
                        (tracked_bbox[0] + tracked_bbox[2], tracked_bbox[1] + tracked_bbox[3]),
                        (255, 255,255),
                        2
                    )
                    cv2.putText(
                        frame,
                        str(person.undetected),
                        (tracked_bbox[0], tracked_bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # Remove persons who have moved left of door from persons_leaving
        # Remove in reverse order to avoid index issues
        for index in sorted(persons_leaving_to_remove, reverse=True):
            del self.persons_leaving[index]
            
        # for index in sorted(persons_to_remove, reverse=True):
        #     del self.persons_inside[index]
           
    # def validate_add(self, results, frame):
    #     """
    #     Validate and add people from YOLO detection results.
    #     Also draws tracking boxes on the frame.
        
    #     Args:
    #         results: YOLO detection results
    #         frame (numpy.ndarray): Current frame for tracker initialization
        
    #     Returns:
    #         list: Temporary array of validated bboxes and tracker features
    #     """
    #     # Create a temporary array to store validated persons
    #     frame2=frame.copy()
    #     validated_data = []
    #     self.check_person_proximity()
    #     # First, process new detections
    #     frame_width = frame.shape[1]
    #     x_threshold = int(frame_width * 0.1)
        
    #     # First, process new detections
    #     for i, result in enumerate(results):
    #         boxes = result.boxes
    #         for j, box in enumerate(boxes):
    #             if int(box.cls[0]) == 0:  # 0 is the index for 'person' in COCO dataset
    #                 x, y, w, h = box.xywh[0].tolist()
                    
    #                 # Calculate the actual left x-coordinate of the box
    #                 left_x = x - w/2
                    
    #                 # Check if left x-coordinate is beyond the 10% threshold
    #                 if left_x >= x_threshold:
    #                     bbox = (int(left_x), int(y-h/2), int(w), int(h))
    #                     validated_data.append({
    #                         'bbox': bbox
    #                     })
    #     # Now check against existing tracked persons
    #     for person in self.persons_inside:
    #         # Extract the person's previous bounding box
    #         x, y, w, h = person.bbox
            
    #         # Crop the portion of the frame corresponding to the previous bbox
           
            
    #         # Calculate feature centroid
    #         centroid_x, centroid_y,_ = self.calculate_feature_centroid(person.image, frame2)
              
    #         # Use centroid to create a new tracked bbox
    #         tracked_bbox = (
    #             int(centroid_x - w/2), 
    #             int(centroid_y - h/2), 
    #             w, 
    #             h
    #         )
            
    #         # Draw rectangle for tracker output (in red)
    #         cv2.rectangle(frame, 
    #                     (int(tracked_bbox[0]), int(tracked_bbox[1])), 
    #                     (int(tracked_bbox[0] + tracked_bbox[2]), 
    #                     int(tracked_bbox[1] + tracked_bbox[3])), 
    #                     (0, 0, 255), 2)  # Red color (BGR)
            
    #         found_matching_bbox = False
            
    #         for entry in validated_data:  # Use copy to avoid modification during iteration
    #             # Calculate IoU between tracked bbox and detected bbox
    #             if self.check_union_area(tracked_bbox, entry['bbox'], 0.7):  # IoU threshold of 0.9
    #                 x, y, w, h = entry['bbox']
    #                 init_rect = (x, y, w, h)
                    
    #                 # Create a new tracker
                    
                    
    #                 # Update person's attributes
    #                 person.bbox = entry['bbox']
    #                 #person.image = frame[y:y+h, x:x+w]  # Crop new image from current frame
    #                 person.inframe = True
    #                 person.exit_flag = 0
    #                 if self.det_close == False and self.det_intersect == False:  
    #                     person.image = frame2[y:y+h, x:x+w]  # Crop new image from current frame

    #                 found_matching_bbox = True
    #                 validated_data.remove(entry)
                    
    #                 # Draw rectangle for matched detection (in green)
    #                 cv2.rectangle(frame, (x, y), 
    #                             (x + w, y + h), 
    #                             (0, 255, 0), 2)  # Green color (BGR)
                    
    #                 break  # Stop searching after finding a match
            
    #         if not found_matching_bbox:
    #             person.inframe = False

    #     persons_to_move = []

    #     for i, person in enumerate(self.persons_inside_room):
    #         # Extract the person's previous bounding box
    #         x, y, w, h = person.bbox
            
    #         # Calculate feature centroid
    #         centroid_x, centroid_y, conf = self.calculate_feature_centroid(person.image, frame2)
            
    #         if conf > 35:
    #             # Use centroid to create a new tracked bbox
    #             tracked_bbox = (
    #                 int(centroid_x - w/2),
    #                 int(centroid_y - h/2),
    #                 w,
    #                 h
    #             )
                
    #             # Draw rectangle for tracker output (in red)
    #             cv2.rectangle(frame,
    #                 (int(tracked_bbox[0]), int(tracked_bbox[1])),
    #                 (int(tracked_bbox[0] + tracked_bbox[2]),
    #                 int(tracked_bbox[1] + tracked_bbox[3])),
    #                 (0,255, 255), 2)  # Red color (BGR)
                    
    #             found_matching_bbox = False
                
    #             for entry in validated_data:  # Use copy to avoid modification during iteration
    #                 # Calculate IoU between tracked bbox and detected bbox
    #                 if self.check_union_area(tracked_bbox, entry['bbox'], 0.7):
    #                     x, y, w, h = entry['bbox']
    #                     init_rect = (x, y, w, h)
                        
    #                     # Update person's attributes
    #                     person.bbox = entry['bbox']
    #                     person.inframe = True
    #                     person.exit_flag = 0
                        
    #                     if self.det_close == False and self.det_intersect == False:
    #                         person.image = frame2[y:y+h, x:x+w]  # Crop new image from current frame
                        
    #                     # Mark this person for moving back to persons_inside
    #                     persons_to_move.append((i, person))
    #                     found_matching_bbox = True
    #                     validated_data.remove(entry)
                        
    #                     # Draw rectangle for matched detection (in green)
    #                     cv2.rectangle(frame, (x, y),
    #                         (x + w, y + h),
    #                         (0, 255, 0), 2)  # Green color (BGR)
    #                     break  # Stop searching after finding a match
                        
    #             if not found_matching_bbox:
    #                 person.inframe = False

    #     # Move persons from persons_inside_room to persons_inside
    #     # Process in reverse order to maintain correct indices
    #     for index, person in sorted(persons_to_move, reverse=True):
    #         # Add to persons_inside
    #         self.persons_inside.append(person)
    #         # Remove from persons_inside_room
    #         del self.persons_inside_room[index]
                    
            
               

    #     # If new detections remain, create Person objects
    #     if len(validated_data) > 0:
    #         print("New untracked persons detected")
    #         for entry in validated_data:
    #             # Get bounding box coordinates
    #             x, y, w, h = entry['bbox']
                
    #             # Convert bbox to format required by tracker
    #             init_rect = (x, y, w, h)
                
    #             # Create a tracker for this person
            
                
    #             # Calculate center of the bounding box
    #             center = (x + w // 2, y + h // 2)
                
    #             # Create a new Person object
    #             new_person = Person(
    #                 bbox=entry['bbox'],
    #                 first_center=center,
    #                 person_id=self.person_counter,
    #                 inframe=True,
    #                 image=frame2[y:y+h, x:x+w]  # Crop image from current frame
    #             )
                
    #             # Check entry of the person
    #             self.check_entry(new_person)
                
    #             # Add to persons_inside
            
                
    #             # Increment person counter
    #             self.person_counter += 1
    # def validate_add(self, results, frame):  confidence light glue 
    #     """
    #     Validate and add people from YOLO detection results.
    #     Also draws tracking boxes on the frame.
        
    #     Args:
    #         results: YOLO detection results
    #         frame (numpy.ndarray): Current frame for tracker initialization
        
    #     Returns:
    #         list: Temporary array of validated bboxes and tracker features
    #     """
    #     # Create a temporary array to store validated persons
    #     frame2=frame.copy()
    #     validated_data = []
    #     self.check_person_proximity()
    #     # First, process new detections
    #     frame_width = frame.shape[1]
    #     x_threshold = int(frame_width * 0.1)
        
    #     # First, process new detections
    #     for i, result in enumerate(results):
    #         boxes = result.boxes
    #         for j, box in enumerate(boxes):
    #             if int(box.cls[0]) == 0:  # 0 is the index for 'person' in COCO dataset
    #                 x, y, w, h = box.xywh[0].tolist()
                    
                    
    #                 bbox = (int(x-w/2), int(y-h/2), int(w), int(h))
    #                 validated_data.append({
    #                     'bbox': bbox
    #                 })
    #     # Now check against existing tracked persons
    #     for person in self.persons_inside:
    #         # Extract the person's previous bounding box
    #         x, y, w, h = person.bbox
            
    #         # Crop the portion of the frame corresponding to the previous bbox
        
            
    #         # Calculate feature centroid
    #         centroid_x, centroid_y,confidence = self.calculate_feature_centroid(person.image, frame2)
    #         found_matching_bbox = False
    #         if confidence > 75:
    #             # Use centroid to create a new tracked bbox
    #             tracked_bbox = (
    #                 int(centroid_x - w/2), 
    #                 int(centroid_y - h/2), 
    #                 w, 
    #                 h
    #             )
                
    #             # Draw rectangle for tracker output (in red)
    #             cv2.rectangle(frame, 
    #                         (int(tracked_bbox[0]), int(tracked_bbox[1])), 
    #                         (int(tracked_bbox[0] + tracked_bbox[2]), 
    #                         int(tracked_bbox[1] + tracked_bbox[3])), 
    #                         (0, 0, 255), 2)  # Red color (BGR)
                
               
                
    #             for entry in validated_data:  # Use copy to avoid modification during iteration
    #                 # Calculate IoU between tracked bbox and detected bbox
    #                 if self.check_union_area(tracked_bbox, entry['bbox'], 0.7):  # IoU threshold of 0.9
    #                     x, y, w, h = entry['bbox']
    #                     init_rect = (x, y, w, h)
                        
    #                     # Create a new tracker
                        
                        
    #                     # Update person's attributes
    #                     person.bbox = entry['bbox']
    #                     #person.image = frame[y:y+h, x:x+w]  # Crop new image from current frame
    #                     person.inframe = True
    #                     person.exit_flag = 0
    #                     if  self.det_intersect == False:  
    #                         person.image = frame2[y:y+h, x:x+w]  # Crop new image from current frame

    #                     found_matching_bbox = True
    #                     validated_data.remove(entry)
                        
    #                     # Draw rectangle for matched detection (in green)
    #                     cv2.rectangle(frame, (x, y), 
    #                                 (x + w, y + h), 
    #                                 (0, 255, 0), 2)  # Green color (BGR)
                        
    #                     break  # Stop searching after finding a match
                
    #         if not found_matching_bbox:
    #             person.inframe = False

        

        
            

    #     # If new detections remain, create Person objects
    #     if len(validated_data) > 0:
    #         print("New untracked persons detected")
    #         for entry in validated_data:
    #             # Get bounding box coordinates
    #             x, y, w, h = entry['bbox']
                
    #             # Convert bbox to format required by tracker
    #             init_rect = (x, y, w, h)
                
    #             # Create a tracker for this person
            
                
    #             # Calculate center of the bounding box
    #             center = (x + w // 2, y + h // 2)
                
    #             # Create a new Person object
    #             new_person = Person(
    #                 bbox=entry['bbox'],
    #                 first_center=center,
    #                 person_id=self.person_counter,
    #                 inframe=True,
    #                 image=frame2[y:y+h, x:x+w]  # Crop image from current frame
    #             )
                
    #             # Check entry of the person
    #             self.check_entry(new_person)
                
    #             # Add to persons_inside
            
                
    #             # Increment person counter
    #             self.person_counter += 1


        
    # def validate_add(self,results, frame):
    #     """
    #     Validate and add people from YOLO detection within door ROI.
    #     Runs YOLO on door area and converts coordinates back to full frame.
        
    #     Args:
    #         frame (numpy.ndarray): Current frame for tracker initialization
            
    #     Returns:
    #         list: Temporary array of validated bboxes and tracker features
    #     """
    #     # Create a temporary array to store validated persons
    #     frame2 = frame.copy()
    #     validated_data = []
    #     self.check_person_proximity()
        
    #     # Draw door region for visualization
    #     door_x, door_y, door_w, door_h = self.door
    #     cv2.rectangle(frame, (door_x, door_y), (door_x + door_w, door_y + door_h), (255, 0, 0), 2)
        
    #     # Process YOLO detections
    #     for i, result in enumerate(results):
    #         boxes = result.boxes
    #         for j, box in enumerate(boxes):
    #             if box.cls.item() == 0:  # Person class
    #                 # Get box coordinates
    #                 x, y, w, h = box.xywh[0].tolist()
    #                 if h > 400 :
    #                     bbox = (int(x-w/2), int(y-h/2), int(w), int(h))
                        
    #                     # Check if detection overlaps with door region
    #                     person_left = bbox[0]
    #                     person_right = bbox[0] + bbox[2]
    #                     door_left = self.door[0]
    #                     door_right = self.door[0] + self.door[2]
                        
    #                     # Only append if person overlaps with door region
    #                     if bbox[0]+(bbox[2]/2) > door_left:
    #                         validated_data.append({
    #                             'bbox': bbox,
    #                             "flag": 0
    #                         })

    #     i = 0
    #     # Now check against existing tracked persons
    #     for person in self.persons_inside:
    #         if person.inframe:
    #             print(i, "oooooooooooiiiiiiiiiiiiiiii")
    #             i += 1
    #             x, y, w, h = person.bbox
    #             tracked_bbox = None
            
    #             # Calculate feature centroid
    #             centroid_x, centroid_y, confidence = self.calculate_feature_centroid(person.image, frame2)
                
    #             if confidence > 35:
    #                 tracked_bbox = (
    #                     int(centroid_x - w/2),
    #                     int(centroid_y - h/2),
    #                     w,
    #                     h
    #                 )

    #             found_matching_bbox = False
    #             if tracked_bbox is not None:
    #                 # Draw rectangle for tracker output (in red)
    #                 cv2.rectangle(frame,
    #                             (int(tracked_bbox[0]), int(tracked_bbox[1])),
    #                             (int(tracked_bbox[0] + tracked_bbox[2]),
    #                             int(tracked_bbox[1] + tracked_bbox[3])),
    #                             (0, 0, 255), 2)
    #                 cv2.putText(frame, f"ID: {person.person_id}", (tracked_bbox[0], tracked_bbox[1] - 10), 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
    #                 for entry in validated_data:
    #                     if entry['flag'] == 0:
    #                         if self.check_union_area(tracked_bbox, entry['bbox'], 0.7):
    #                             x, y, w, h = entry['bbox']
                                
    #                             # Update entry and person
    #                             entry['flag'] = 1
    #                             print(person.person_id, "i dddddddddddddddddddd")
    #                             #$if not person.tracking:
    #                             person.bbox = entry['bbox']
    #                             person.inframe = True
    #                             person.exit_flag = 0
    #                             if self.det_intersect == False:
    #                                 person.image = frame2[y:y+h, x:x+w]
    #                             # else : 
    #                             #     entry_x, entry_y, _, _ = entry['bbox']
    #                             #     _, _, tracked_w, tracked_h = tracked_bbox
                                    
    #                             #     # Combine coordinates
    #                             #     person.bbox = (entry_x, entry_y, tracked_w, tracked_h)
                                    
    #                             #     # Update person properties
    #                             #     person.inframe = True
    #                             #     person.exit_flag = 0
                                    
    #                             #     # Get image using new combined coordinates
    #                             #     if self.det_intersect == False:
    #                             #         person.image = frame2[entry_y:entry_y+tracked_h, entry_x:entry_x+tracked_w]
    #                             found_matching_bbox = True
    #                             validated_data.remove(entry)
                                
    #                             # Draw rectangle for matched detection (in green)
    #                             cv2.rectangle(frame, (x, y),
    #                                         (x + w, y + h),
    #                                         (0, 255, 0), 2)
    #                             cv2.putText(frame, f"Matched ID: {person.person_id}", (x, y - 10), 
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
    #                             break
    #     for person in self.persons_leaving:
    #         if person.inframe:
    #             print(i, "oooooooooooiiiiiiiiiiiiiiii")
    #             i += 1
    #             x, y, w, h = person.bbox
    #             tracked_bbox = None
            
    #             # Calculate feature centroid
    #             centroid_x, centroid_y, confidence = self.calculate_feature_centroid(person.image, frame2)
                
    #             if confidence > 35:
    #                 tracked_bbox = (
    #                     int(centroid_x - w/2),
    #                     int(centroid_y - h/2),
    #                     w,
    #                     h
    #                 )

    #             found_matching_bbox = False
    #             if tracked_bbox is not None:
    #                 # Draw rectangle for tracker output (in red)
    #                 cv2.rectangle(frame,
    #                             (int(tracked_bbox[0]), int(tracked_bbox[1])),
    #                             (int(tracked_bbox[0] + tracked_bbox[2]),
    #                             int(tracked_bbox[1] + tracked_bbox[3])),
    #                             (0, 0, 255), 2)
    #                 cv2.putText(frame, f"ID: {person.person_id}", (tracked_bbox[0], tracked_bbox[1] - 10), 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
    #                 for entry in validated_data:
    #                     if entry['flag'] == 0:
    #                         if self.check_union_area(tracked_bbox, entry['bbox'], 0.5):
    #                             x, y, w, h = entry['bbox']
                                
    #                             # Update entry and person
    #                             entry['flag'] = 1
    #                             print(person.person_id, "i dddddddddddddddddddd")
    #                             #$if not person.tracking:
    #                             person.bbox = entry['bbox']
    #                             person.inframe = True
    #                             person.exit_flag = 0
    #                             if self.det_intersect == False:
    #                                 person.image = frame2[y:y+h, x:x+w]
    #                             # else : 
    #                             #     entry_x, entry_y, _, _ = entry['bbox']
    #                             #     _, _, tracked_w, tracked_h = tracked_bbox
                                    
    #                             #     # Combine coordinates
    #                             #     person.bbox = (entry_x, entry_y, tracked_w, tracked_h)
                                    
    #                             #     # Update person properties
    #                             #     person.inframe = True
    #                             #     person.exit_flag = 0
                                    
    #                             #     # Get image using new combined coordinates
    #                             #     if self.det_intersect == False:
    #                             #         person.image = frame2[entry_y:entry_y+tracked_h, entry_x:entry_x+tracked_w]
    #                             found_matching_bbox = True
    #                             validated_data.remove(entry)
                                
    #                             # Draw rectangle for matched detection (in green)
    #                             cv2.rectangle(frame, (x, y),
    #                                         (x + w, y + h),
    #                                         (0, 255, 0), 2)
    #                             cv2.putText(frame, f"Matched ID: {person.person_id}", (x, y - 10), 
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
    #                             break
    #     # Process new detections
    #     if len(validated_data) > 0:
    #         print("New untracked persons detected")
    #         for entry in validated_data:
    #             x, y, w, h = entry['bbox']
    #             center = (x + w // 2, y + h // 2)
                
    #             new_person = Person(
    #                 bbox=entry['bbox'],
    #                 first_center=center,
    #                 person_id=self.person_counter,
    #                 inframe=True,
    #                 image=frame2[y:y+h, x:x+w]
    #             )
                
    #             # Check if person is entering through left half of door
    #             if x+ w/2 < (door_x + door_w/2):
    #                 self.persons_inside.append(new_person)
    #             else :
    #                 self.persons_leaving.append(new_person)
    #             self.person_counter += 1
    def validate_add(self, online_targets, frame):
        """
        Validate and add people based on tracking IDs without YOLO detection.
        Tracks people through temporary list and detection counts.
        
        Args:
            online_targets: List of tracked targets with IDs and bounding boxes
            frame: Current frame for visualization
        """
        # Initialize lists if they don't exist
        if not hasattr(self, 'temp_list'):
            self.temp_list = []
            
        # Draw door region for visualization
        door_x, door_y, door_w, door_h = self.door
        cv2.rectangle(frame, (door_x, door_y), (door_x + door_w, door_y + door_h), (255, 0, 0), 2)
        
        # Process online targets
        for target in online_targets:
            tlwh = target.tlwh
            tid = target.track_id
            
            # Extract width and height from tlwh
            width = tlwh[2]   
            height = tlwh[3]  
            
            # Check if target meets height and width criteria
            if height > 400 and width > 150:
                # Extract image region for the person
                x, y = int(tlwh[0]), int(tlwh[1])
                w, h = int(tlwh[2]), int(tlwh[3])
                person_region = frame[y:y+h, x:x+w]
            else:
                continue
                
            # Calculate center point
            center_x = tlwh[0] + tlwh[2] / 2
            center_y = tlwh[1] + tlwh[3] / 2
            center = (center_x, center_y)
            
            # Check if this ID exists in temp_list
            existing_temp = next((p for p in self.temp_list if p.person_id == tid), None)
            
            if not existing_temp:
                # Create new person object
                new_person = Person(
                    bbox=tlwh,
                    first_center=center,
                    person_id=tid,
                    image = person_region,
                    inframe=True
                )
                new_person.det_count = 1
                  # Save the image region
                self.temp_list.append(new_person)
                
                # Draw visualization for new detection
                cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), 
                            (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), 
                            (255, 255, 0), 2)
                cv2.putText(frame, f"New ID: {tid}", (int(tlwh[0]), int(tlwh[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                # Update existing person in temp_list
                existing_temp.bbox = tlwh
                existing_temp.image = person_region  # Update the image region
                existing_temp.det_count += 1
                
                # Check if detection count threshold is reached
                if existing_temp.det_count >= 5:
                    # Determine if person is inside door region
                    first_center_x = existing_temp.first_center[0]
                    is_in_door = (door_x <= first_center_x <= door_x + door_w)
                    
                    # Check if person already exists in either list
                    exists_inside = any(p.person_id == tid for p in self.persons_inside)
                    exists_leaving = any(p.person_id == tid for p in self.persons_leaving)
                    
                    if not exists_inside and not exists_leaving:
                        if is_in_door:
                            self.persons_inside.append(existing_temp)
                            print(f"Person {tid} added to persons_inside")
                        else:
                            self.persons_leaving.append(existing_temp)
                            print(f"Person {tid} added to persons_leaving")
                        
                        # Remove from temp_list after assignment
                        self.temp_list.remove(existing_temp)
                
                # Draw visualization for existing detection
                cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), 
                            (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), 
                            (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {tid} (Count: {existing_temp.det_count})", 
                        (int(tlwh[0]), int(tlwh[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update tracked persons in both lists
        for person_list in [self.persons_inside, self.persons_leaving]:
            for person in person_list:
                # Check if person's ID exists in current targets
                matching_target = next((t for t in online_targets if t.track_id == person.person_id), None)
                if matching_target:
                    person.bbox = matching_target.tlwh
                    # Update person's image
                    x, y = int(matching_target.tlwh[0]), int(matching_target.tlwh[1])
                    w, h = int(matching_target.tlwh[2]), int(matching_target.tlwh[3])
                    person.image = frame[y:y+h, x:x+w]
                    person.inframe = True
                    person.undetected = 0
                else:
                    person.undetected += 1
    
    def check_exit(self):
        """
        Check for people who have left through the door.
        - If a person is not in frame and their bbox is near the door,
        remove them from persons_leaving and remove one person from persons_inside
        - If a person is not in frame but not near door, only remove from persons_leaving
        """
        # Create a list to store indices of people to remove from persons_leaving
        to_remove_leaving = []

        # Go through all persons in persons_leaving list
        for i, person in enumerate(self.persons_leaving):
            # Check if person is marked as not in frame and has a valid bbox
            if person.inframe is False and person.bbox is not None:
                # Remove person from persons_leaving
                to_remove_leaving.append(i)
                
                # If their last position was near the door, also remove one person from persons_inside
                if self.is_bbox_close(person.bbox, self.door) and len(self.persons_inside) > 0:
                    # Remove the first person from persons_inside (index doesn't matter)
                    self.persons_inside.pop(0)

        # Remove marked persons from persons_leaving in reverse order
        for index in sorted(to_remove_leaving, reverse=True):
            del self.persons_leaving[index]
    def is_bbox_close(self, bbox1, bbox2, threshold=100):
        """
        Check if two bounding boxes are close to each other.
        
        Args:
        bbox1 (tuple): First bounding box (x, y, w, h)
        bbox2 (list/tuple): Second bounding box 
        threshold (int): Maximum allowed distance between bbox centers
        
        Returns:
        bool: True if bboxes are close, False otherwise
        """
        # Extract center coordinates of first bbox
        x1, y1, w1, h1 = bbox1
        center1_x = x1 + w1 // 2
        center1_y = y1 + h1 // 2
        
        # Extract center coordinates of second bbox
        # Handle potential difference in input format
        if isinstance(bbox2, list):
            x2, y2, w2, h2 = bbox2
        else:
            x2, y2, w2, h2 = bbox2 
        
        center2_x = x2 + int(w2) // 2
        center2_y = y2 + int(h2) // 2
        
        # Calculate Euclidean distance
        distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
        
        # Return True if distance is within threshold
        return distance <= threshold

        

    def get_person_by_id(self, person_id):
        """
        Retrieve a Person object by its ID.
        
        Args:
            person_id (int): ID of the person to find
        
        Returns:
            Person or None: The Person object if found, else None
        """
        for person in self.persons:
            if person.person_id == person_id:
                return person
        return None
    
    def check_entry(self, person, threshold=100):
        """
        Check if the first detected center of a person is close to the door's center.
        Args:
        person (Person): Person object to check
        threshold (int, optional): Maximum allowed distance in pixels. Defaults to 100.
        Returns:
        bool: True if the person's first center is within the threshold of the door's center, False otherwise
        """
        # Unpack the first detected center of the person
        person_center_x, person_center_y = person.first_center
        
        # Calculate the door's center from its bounding box
        # Assuming self.door is in the format [x, y, w, h]
        door_x, door_y, door_w, door_h = self.door
        door_center_x = door_x + door_w // 2
        door_center_y = door_y + door_h // 2
        
        # Calculate the absolute difference between centers
        dx = abs(person_center_x - door_center_x)
        dy = abs(person_center_y - door_center_y)
        
        # Calculate the Euclidean distance
        distance = (dx**2 + dy**2)**0.5
        
        # Check if the distance is within the threshold
        if distance <= threshold:
            # If within threshold, add the person to the inside list
            self.add_person_inside(person)
           
    
    def __len__(self):
        """
        Return the number of persons in the list.
        """
        return len(self.persons)
    
    def __repr__(self):
        """
        String representation of the PersonList.
        """
        return f"PersonList(persons={self.persons})"