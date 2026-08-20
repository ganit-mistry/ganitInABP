import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from Classes.imageurl import *

class MainFace:

    def __init__(self, image):
        self.image = image 
        self.yolo_model = YOLO('yolov8n.pt')

    def load_model_and_infer(self):
        '''Loads YOLO and InsightFace models'''
        # insightface_model = FaceAnalysis(name='buffalo_l')
        # insightface_model.prepare(ctx_id=-1)  # -1 for CPU and 0 for GPU

        persons = self.yolo_model(self.image)
        # faces = insightface_model.get(self.image)

        return persons

    def find_largest_face(self):
        '''Finds the largest face in the image'''
        _, faces = self.load_model_and_infer()

        faces_dictionary = {}
        for idx, face in enumerate(faces):
            faces_dictionary[idx] = [float(face.bbox[0]), float(face.bbox[1]), float(face.bbox[2]), float(face.bbox[3])]

        areas = {}
        for key, coords in faces_dictionary.items():
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1
            areas[key] = width * height

        max_face_area_val = max(areas.values())
        max_face_area_val_key = max(areas, key=areas.get)

        return max_face_area_val, max_face_area_val_key

    def find_person_for_largest_face(self,face_coords):
        '''Finds the person corresponding to the largest face'''
        persons, faces = self.load_model_and_infer()

        persons_dictionary = {}
        for idx, person in enumerate(persons[0].boxes):
            if person.cls == 0:
                
                persons_dictionary[idx] = [
                    person.xyxy[0][0].item(),
                    person.xyxy[0][1].item(),
                    person.xyxy[0][2].item(),
                    person.xyxy[0][3].item()
                ]

        faces_dictionary = {}
        for idx, face in enumerate(faces):
            faces_dictionary[idx] = [
                float(face.bbox[0]), float(face.bbox[1]), float(face.bbox[2]), float(face.bbox[3])
            ]

        max_face_area_val, max_face_area_val_key = self.find_largest_face()

        faces_dictionary_max = faces_dictionary[max_face_area_val_key]

        for id in persons_dictionary:
            if faces_dictionary_max[0] >= persons_dictionary[id][0] and faces_dictionary_max[1] >= persons_dictionary[id][1] and faces_dictionary_max[2] <= persons_dictionary[id][2] and faces_dictionary_max[3] <= persons_dictionary[id][3]:
                persons_dictionary_max = persons_dictionary[id]

        return faces_dictionary_max, persons_dictionary_max


    def rohit_da_code(self,face_coords):
        '''Finds the person corresponding to the largest face'''
        persons = self.load_model_and_infer()
        persons_dictionary = {}
        for idx, person in enumerate(persons[0].boxes):
            if person.cls == 0:
                # person_crop = self.image[person.xyxy[0][3]:person.xyxy[0][2],person.xyxy[0][1]:person.xyxy[0][0]]
                persons_dictionary[idx] = [
                    person.xyxy[0][0].item(),
                    person.xyxy[0][1].item(),
                    person.xyxy[0][2].item(),
                    person.xyxy[0][3].item()
                ]

        faces_dictionary_max = face_coords[0]

        persons_dictionary_max = persons_dictionary[0]
        for id in persons_dictionary:
            if faces_dictionary_max[0] >= persons_dictionary[id][0] and faces_dictionary_max[1] >= persons_dictionary[id][1] and faces_dictionary_max[2] <= persons_dictionary[id][2] and faces_dictionary_max[3] <= persons_dictionary[id][3]:
                persons_dictionary_max = persons_dictionary[id]
        return faces_dictionary_max, persons_dictionary_max

    def face_percent(self,face_coords):
        '''Calculates the face area percentage over the person area'''
        face, person = self.rohit_da_code(face_coords)
        face_area, person_area = 0, 0

        face_width = face[2] - face[0]
        face_height = face[3] - face[1]
        face_area = face_width * face_height

        person_width = person[2] - person[0]
        person_height = person[3] - person[1]
        person_area = person_width * person_height

        percent = (face_area / person_area) * 100
        return percent


class SegmentationProcessor:
    def __init__(self, model_name="mattmdjaga/segformer_b2_clothes"):
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

    def apply_segmentation(self, image: Image, bbox: list):
        '''Crops the image based on the provided bbox and applies Segformer segmentation'''        
        left, top, right, bottom = map(int, bbox)
        cropped_image = image.crop((left, top, right, bottom))

        inputs = self.processor(images=cropped_image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=cropped_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        return pred_seg, cropped_image

    def calculate_skin_and_dress_area(self, pred_seg):
        '''Calculate the skin and dress area based on the segmentation'''

        global skin_percentage, dress_percentage

        id_to_label = {
            0: "Background",
            1: "Hat",
            2: "Hair",
            3: "Sunglasses",
            4: "Upper-clothes", 
            5: "Skirt",
            6: "Pants",
            7: "Dress",
            8: "Belt",
            9: "Left-shoe", 
            10: "Right-shoe",
            11: "Face",
            12: "Left-leg",
            13: "Right-leg",
            14: "Left-arm", 
            15: "Right-arm",
            16: "Bag",
            17: "Scarf"
        }

        unique_ids = torch.unique(pred_seg)
        total_area = pred_seg.numel()

        skin_area = 0
        dress_area = 0
    
        for seg_id in unique_ids:
            label = id_to_label.get(seg_id.item(), "Unknown")
            area = torch.sum(pred_seg == seg_id).item()

            if seg_id.item() in [11, 12, 13, 14, 15]:   # IDs related to skin
                skin_area += area
            if seg_id.item() in [4, 5, 6, 7]:           # IDs related to dress
                dress_area += area

        skin_percentage = (skin_area / total_area) * 100
        dress_percentage = (dress_area / total_area) * 100
        # globals["sp"] = skin_percentage
        # globals["dp"] = dress_percentage

        return skin_percentage, dress_percentage


    def process_image_from_url(self,image_url,face_coords):
        # from imageurl import ImageURL  # Import the ImageURL class here
        
        # Get the image as numpy array using ImageURL
        image_url_processor = ImageURL()
        image_data, error = image_url_processor.image_url_to_array(image_url)
        
        if error:
            return f"Error: {error}"

        # Initialize processors
        face_processor = MainFace(image_data)
        print("before Try")

        try:
            print("In try")
            largest_face_coords, largest_person_coords = face_processor.rohit_da_code(face_coords)
            
            # Segmentation processing
            seg_processor = SegmentationProcessor()
            pred_seg, _ = seg_processor.apply_segmentation(Image.fromarray(image_data), largest_person_coords)
            
            skin_percentage, dress_percentage = seg_processor.calculate_skin_and_dress_area(pred_seg)
            
            face_percentage = face_processor.face_percent(face_coords)

            # Status
            if skin_percentage < dress_percentage:
                acceptance_status = "Accepted"
            elif skin_percentage > dress_percentage:
                acceptance_status = "Rejected"
            else:
                acceptance_status = "Error"

            print(f"Skin Percentage: {skin_percentage:.2f}%", 
                    f"Dress Percentage: {dress_percentage:.2f}%", 
                    f"Face Area Percentage in Person: {face_percentage:.2f}%", 
                    acceptance_status)
        
            return acceptance_status

        except Exception as e:
            return f"Error: {str(e)}", "", "", "Error"


# Example usage
# image_url = input("Enter the URL of the image: ")  # Example: "https://example.com/image.jpg"
# result = process_image_from_url(image_url)
# for r in result:
#     print(r)
