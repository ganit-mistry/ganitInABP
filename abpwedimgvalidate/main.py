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
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# INSIGHT-FACE
try:
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
except Exception as e:
    print(f"(33) Error Loading InsightFace Model: {e}")
    app = None

# List to store the results
result_data = []

def return_status(status, idresult=[], confidence_score={}, confidence=None):
    ID_1 = None
    ID_2 = None
    ID_3 = None
    ID_4 = None
    ID_5 = None
    ID_6 = None
    ID_7 = None
    ID_8 = None

    if 'ID_1' in idresult:
        ID_1 = 1
    if 'ID_2' in idresult:
        ID_2 = 1
    if 'ID_3' in idresult:
        ID_3 = 1
    if 'ID_4' in idresult:
        ID_4 = 1
    if 'ID_5' in idresult:
        ID_5 = 1
    if 'ID_6' in idresult:
        ID_6 = 1
    if 'ID_7' in idresult:
        ID_7 = 1
    if 'ID_8' in idresult:
        ID_8 = 1

    return {
        "status": status,
        "DetectedClass": {
            "ID_1": ID_1,
            "ID_2": ID_2,
            "ID_3": ID_3,
            "ID_4": ID_4,
            "ID_5": ID_5,
            "ID_6": ID_6,
            "ID_7": ID_7,
            "ID_8": ID_8
        },
        "confidence_scores": confidence_score
    }

def save_results_to_csv(results, filename="results.csv"):
    """ Save the results to a CSV file """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def get_result(image_url):
    global result_data  # Use the global list to store results
    Face_detecttion = FaceDetecttion()
    final_result = ""
    confidence_scores = {}
    status = 0 

    # ImageURL
    Image_Url = ImageURL()
    image, error = Image_Url.image_url_to_array(image_url)
    if error:
        result_data.append({
            "ImageURL": image_url,
            "Status": "Error",
            "IDs": "ID_1",
            "Percentage": f"SP:{globals['sp']} and dp:{globals['dp']}"
        })
        save_results_to_csv(result_data)  # Save after error
        return return_status(status, ['ID_1'])  # Error in image URL

    # Processing NSFW
    Detect_Nsfw = DetectNSFW()
    NSFW_String, NSFW_Confidence = Detect_Nsfw.detect_nsfw(image)
    
    if NSFW_String == "Image contains NSFW content":
        result_data.append({
            "ImageURL": image_url,
            "Status": "NSFW",
            "IDs": "ID_2"
        })
        save_results_to_csv(result_data)  # Save after NSFW detection
        return return_status(status, ['ID_2'], confidence_score={"NSFW": NSFW_Confidence}, confidence=NSFW_Confidence)

    # Semi Nude
    face_for_semi_nude = Face_detecttion.crop_faces_for_nsfw(image)
    Segmentation_Processor = SegmentationProcessor()
    acceptance_status = Segmentation_Processor.process_image_from_url(image_url,face_for_semi_nude)

    if acceptance_status == "Rejected":
        result_data.append({
            "ImageURL": image_url,
            "Status": "Rejected",
            "IDs": "ID_8"
        })
        save_results_to_csv(result_data)  # Save after Rejected
        return return_status(status, ['ID_8'], confidence_score={"Status": "Rejected"}, confidence=None)

    # POCR    
    P_Ocr = Pocr()
    Phone_Number_Result = P_Ocr.pocr(image)
    if Phone_Number_Result == "Rejected":
        result_data.append({
            "ImageURL": image_url,
            "Status": "Rejected",
            "IDs": "ID_7"
        })
        save_results_to_csv(result_data)  # Save after POCR rejection
        return return_status(status,['ID_7'])

    #  ANIMATED IMAGES
    Animated_image = Animatedimage()
    Cartoon_Face_Result, Error_Code = Animated_image.check_if_cartoon(image)
    if Cartoon_Face_Result == "Cartoon":
        result_data.append({
            "ImageURL": image_url,
            "Status": "Cartoon Detected",
            "IDs": "ID_3"
        })
        save_results_to_csv(result_data)  # Save after Cartoon detection
        return return_status(status,['ID_3'])

    # Face Detection
    
    Face_Result, Error_Code = Face_detecttion.check_image(image, app)

    if Face_Result == "Rejected":    
        if Error_Code == 0:
            result_data.append({
                "ImageURL": image_url,
                "Status": "Face Detected",
                "IDs": "ID_3"
            })
        elif Error_Code == 1:
            result_data.append({
                "ImageURL": image_url,
                "Status": "Face Detected with Error Code 1",
                "IDs": "ID_4"
            })
        else:
            result_data.append({
                "ImageURL": image_url,
                "Status": "Face Detected with Error",
                "IDs": "ID_1"
            })
        save_results_to_csv(result_data)  # Save after face detection
        return return_status(status, ['ID_1'], {})

    # MediaPipe CLIP YOLO
    Clip_yolo = MediaPipeClipYolo()
    Result2, errormedia = Clip_yolo.process_single_image(image)
    Result3, errorclip, clip_confidence, detected_class = Clip_yolo.process_image_clip(image)
    Result4, erroryolo, yolo_confidence, yolo_class = Clip_yolo.process_yolo(image)

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
        final_result = return_status(1)  
    elif errorclip is None and erroryolo == "sunglasses":
        final_result = return_status(1)
    else:
        final_result = return_status(0, ['ID_5', 'ID_6'], confidence_scores)

    result_data.append({
        "ImageURL": image_url,
        "Status": "Final Result",
        "IDs": ",".join([str(key) for key, val in final_result["DetectedClass"].items() if val])
    })

    # Save results after processing all steps
    save_results_to_csv(result_data)

    # Combined Result
    print("\n\nCOMBINED RESULT:")
    print(f" \n Insight Face Result: {Face_Result}, \n Media pipe Result: {Result2}, \n Clip B/32 Result: {Result3}, \n yolo Result: {Result4}, \n")
    print(f"------------------------------------------------------------------------------------------------------------------------------------")
    
    '''
    # Processing YOLO
    YOLO_Processor = YOLO()
    yolo_status, detected_class, conf, _ = YOLO_Processor.process_yolo(image)

    if detected_class and detected_class != "eyeglasses":
        return return_status(status, ['ID_5', 'ID_6'], confidence_score={"YOLO": conf}, confidence=conf) 
    '''
    
    return final_result
    #return return_status(status)  # No issues detected
