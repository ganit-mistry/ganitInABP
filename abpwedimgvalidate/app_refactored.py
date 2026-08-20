import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException,Depends,File, UploadFile
from pydantic import BaseModel
from typing import Optional
import os
import sys
import pandas as pd  # Import pandas for handling CSV/Excel
from insightface.app import FaceAnalysis

from Classes.imageurl import ImageURL
from Classes.detectnsfw import DetectNSFW
from Classes.pocr import Pocr
from Classes.detectanime import Animatedimage
from Classes.detectface import FaceDetecttion
from Classes.mediaclipyolo import MediaPipeClipYolo

from Classes.detectseminude import *

import warnings


app = FastAPI()


# Define request body model
#class ImageData(BaseModel):
 #   image_url: Optional[str] = None


# Define request body model
class ImageRequest:
    def __init__(self):
        self.image_url_parser = ImageURL()
        self.nsfw_detector = DetectNSFW()
        self.ocr_engine = Pocr()
        self.gif_checker = Animatedimage()
        self.face_detector = FaceDetecttion()
        self.pose_detector = MediaPipeClipYolo()
        self.segmentation_processor = SegmentationProcessor()

        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1)

        self.result_data = []
    def return_status(self, status, idresult=None, confidence_score=None, confidence=None):
        if idresult is None:
            idresult = []
        if confidence_score is None:
            confidence_score = {}

        detected = {f"ID_{i}": (1 if f"ID_{i}" in idresult else None) for i in range(1, 9)}

        return {
            "status": status,
            "DetectedClass": detected,
            "confidence_scores": confidence_score
        }

    
    def save_results_to_csv(self, results, filename="results.csv"):
        """ Save the results to a CSV file """
        if not results or not isinstance(results, list):
            print("Warning: No valid results to save.")
            return

        try:
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")


    def get_result(self,image_url):
        final_result = ""
        confidence_scores = {}
        status = 0 

        # ImageURL
        image, error = self.image_url_parser.image_url_to_array(image_url)
        if error:
            self.result_data.append({
                "ImageURL": image_url,
                "Status": "Error",
                "IDs": "ID_1",
                "Percentage": f"SP:{globals['sp']} and dp:{globals['dp']}"
            })
            ##self.save_results_to_csv(self.result_data)  # Save after error
            return self.return_status(status, ['ID_1'])  # Error in image URL
        
        # Processing NSFW

        NSFW_String, NSFW_Confidence = self.nsfw_detector.detect_nsfw(image)
        if NSFW_String == "Image contains NSFW content":
            self.result_data.append({
                "ImageURL": image_url,
                "Status": "NSFW",
                "IDs": "ID_2"
            })
            ##self.save_results_to_csv(self.result_data)  # Save after NSFW detection
            return self.return_status(status, ['ID_2'], confidence_score={"NSFW": NSFW_Confidence}, confidence=NSFW_Confidence)
        
        # Semi Nude
        face_for_semi_nude = self.face_detector.crop_faces_for_nsfw(image)
        
        print("A")
        acceptance_status = self.segmentation_processor.process_image_from_url(image_url,face_for_semi_nude)
        print("B")

        if acceptance_status == "Rejected":
            self.result_data.append({
                "ImageURL": image_url,
                "Status": "Rejected",
                "IDs": "ID_8"
            })
            #self.save_results_to_csv(self.result_data)  # Save after Rejected
            return self.return_status(status, ['ID_8'], confidence_score={"Status": "Rejected"}, confidence=None)
        
        # POCR

        Phone_Number_Result = self.ocr_engine.pocr(image)
        if Phone_Number_Result == "Rejected":
            self.result_data.append({
                "ImageURL": image_url,
                "Status": "Rejected",
                "IDs": "ID_7"
            })
            #self.save_results_to_csv(self.result_data)  # Save after POCR rejection
            return self.return_status(status,['ID_7'])
        
        #  ANIMATED IMAGES

        Cartoon_Face_Result, Error_Code = self.gif_checker.check_if_cartoon(image)
        if Cartoon_Face_Result == "Cartoon":
            self.result_data.append({
                "ImageURL": image_url,
                "Status": "Cartoon Detected",
                "IDs": "ID_3"
            })
            #self.save_results_to_csv(self.result_data)  # Save after Cartoon detection
            return self.return_status(status,['ID_3'])
        
        Face_Result, Error_Code = self.face_detector.check_image(image)

        if Face_Result == "Rejected":    
            if Error_Code == 0:
                self.result_data.append({
                    "ImageURL": image_url,
                    "Status": "Face Detected",
                    "IDs": "ID_3"
                })
                return self.return_status(status, ['ID_3'], {})
                
            elif Error_Code == 1:
                self.result_data.append({
                    "ImageURL": image_url,
                    "Status": "Face Detected with Error Code 1",
                    "IDs": "ID_4"
                })
                return self.return_status(status, ['ID_4'], {})
            else:
                self.result_data.append({
                    "ImageURL": image_url,
                    "Status": "Face Detected with Error",
                    "IDs": "ID_1"
                })
                return self.return_status(status, ['ID_1'], {})
        
        # MediaPipe CLIP YOLO
        
        Result2, errormedia = self.pose_detector.process_single_image(image)
        Result3, errorclip, clip_confidence, detected_class = self.pose_detector.process_image_clip(image)
        Result4, erroryolo, yolo_confidence, yolo_class = self.pose_detector.process_yolo(image)

        clip_confidence = float(clip_confidence) if clip_confidence is not None else 0.0
        yolo_confidence = float(yolo_confidence) if yolo_confidence is not None else 0.0

        confidence_scores['CLIP B32'] = {
            "Confidence": clip_confidence,
            "Detected Class": detected_class
        }
        
        confidence_scores['YOLO'] = {
            "Confidence": yolo_confidence,
            "Detected Class": yolo_class
        }

        accepted_count = sum([Result2 == 'Accepted', Result3 == 'Accepted', Result4 == 'Accepted'])
        
        if accepted_count >= 2:
            final_result = self.return_status(1)  
        elif errorclip is None and erroryolo == "sunglasses":
            final_result = self.return_status(1)
        else:
            final_result = self.return_status(0, ['ID_5', 'ID_6'], confidence_scores)

        self.result_data.append({
            "ImageURL": image_url,
            "Status": "Final Result",
            "IDs": ",".join([str(key) for key, val in final_result["DetectedClass"].items() if val])
        })

        # Save results after processing all steps
        #self.save_results_to_csv(self.result_data)
        return final_result    
def get_image_request():
    return ImageRequest()

executor = ThreadPoolExecutor(max_workers=5)

@app.post("/process_image")
async def process_image(file: UploadFile = File(...),request_processor: ImageRequest=Depends(get_image_request)):
    #if not request_data.image_url:
     #   raise HTTPException(status_code=400, detail="Image URL is required")
    
    loop = asyncio.get_running_loop()
    # Run get_result in executor to avoid blocking
    result = await loop.run_in_executor(executor, request_processor.get_result, file)
    return result
