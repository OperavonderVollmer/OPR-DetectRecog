from OPRDetectRecog.Interfaces.Detector import Detector
from OPRDetectRecog.Interfaces.Recognizer import Recognizer
from OperaPowerRelay import opr
from PIL import Image

import os, importlib.util
from pathlib import Path

def load_detectors(specific_detector: str = None) -> Detector | dict[str, Detector]:


    """
    Loads detector modules in the "Detectors" directory and returns their instances.

    If a specific detector is specified, it will be returned as a Detector instance.
    Otherwise, a dictionary of all detectors will be returned.

    Parameters
    ----------
    specific_detector : str, optional
        The name of a specific detector to load.

    Returns
    -------
    Detector | dict[str, Detector]
        A Detector instance or a dictionary of Detector instances, depending on the input.
    """

    detectors_path = Path(__file__).resolve().parent / "Detectors"

    detectors = {}

    for file_name in os.listdir(detectors_path):
        if not file_name.endswith(".py"):
            continue

        detector_name = file_name[:-3]
        detector_path = os.path.join(detectors_path, file_name)


        spec = importlib.util.spec_from_file_location(detector_name, detector_path)

        det = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(det)

        if specific_detector is None and hasattr(det, "get_detector"):
            detectors[detector_name] = det.get_detector()

        if detector_name == specific_detector:
            return det.get_detector()

    return detectors


def load_recognizers(specific_recognizer: str = None) -> Recognizer | dict[str, Recognizer]:

    """
    Loads recognizer modules in the "Recognizers" directory and returns their instances.

    If a specific recognizer is specified, it will be returned as a Recognizer instance.
    Otherwise, a dictionary of all recognizers will be returned.

    Parameters
    ----------
    specific_recognizer : str, optional
        The name of a specific recognizer to load.

    Returns
    -------
    Recognizer | dict[str, Recognizer]
        A Recognizer instance or a dictionary of Recognizer instances, depending on the input.
    """
    
    recognizers_path = Path(__file__).resolve().parent / "Recognizers"

    recognizers = {}

    for file_name in os.listdir(recognizers_path):
        if not file_name.endswith(".py"):
            continue

        recognizer_name = file_name[:-3]
        recognizer_path = os.path.join(recognizers_path, file_name)


        spec = importlib.util.spec_from_file_location(recognizer_name, recognizer_path)

        rec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rec)

        if specific_recognizer is None and hasattr(rec, "get_recognizer"):
            recognizers[recognizer_name] = rec.get_recognizer()

        if recognizer_name == specific_recognizer:
            return rec.get_recognizer()

    return recognizers



def main(p: str = None) -> None:

    
    """

        OPR Detect Recog v1.0.0

        A quick demonstration of how to use OPR Detect Recog

        After loading the recognizers and detectors, you have to initialize them

        Recognizers initialize using the language, tolerance, and path (note that some of them don't use these, just check whichever you're looking to use)

        Detectors initialize using the language and path (same case, but language is, for the most part, a requirement here, at least for the single detector we currently use)

        Detector uses an image and converts it to a numpy array first before processing. From a single image it returns a list of its results, the point is to detect multiple text objects in an image. The result is a list of QuadBox objects, numpy array, and pil images, to ensure that its compatible with any recognizer. Make sure to pass the quadbox to the recognizer to package it into a linguist result object

        Recognizer can use either an image or a numpy array depending on the recognizer you're using. It works one by one, line by line since working with a whole list might not work for certain use cases, might change that in the future #TODO
        It returns the results as a list of LinguistResult objects, the purpose is to get the recognized text and the confidence level, along with the bounding box of where, in the original image, did the text originally appear from.

        From here, you're free to use the results however you want.

        NOTE: I am currently using a AMD GPU so I do not have CUDA, therefore I cannot properly test the performance and speeds of the detectors and recognizers. From my experience, paddleocr_detector + mangaocr_recognizer is the quickest and most accurate combo, with paddleocr_detector + easyocr_recognizer being slightly worse. Paddleocr_detector+recognizer is easily the fastest and least accurate for my machine. Willing to test more if someone would generously gift me a RTX 3080 or something :) <3
    
    """


    print("OPR Detect Recog v1.0.0 demo")


    recognizers = load_recognizers()
    detectors = load_detectors()


    while True:
        opr.list_choices(detectors.keys(), "Select a Detector")

        index_detector = opr.input_from("OPR Detector", "Select a Detector", 1)
        
        try:
            detector_name = list(detectors.keys())[int(index_detector) -1] 
            detector = detectors[detector_name]

            opr.print_from("OPR Detector", f"Selected Detector {detector.Name}")
            break

        except (KeyError, ValueError, IndexError):
            opr.print_from("OPR Detector", "Detector not found!")
            continue

    while True:
        opr.list_choices(recognizers.keys(), "Select a Recognizer")

        index_recognizer = opr.input_from("OPR Recognizer", "Select a Recognizer", 1)
        
        try:
            recognizer_name = list(recognizers.keys())[int(index_recognizer) -1] 
            recognizer = recognizers[recognizer_name]

            opr.print_from("OPR Recognizer", f"Selected Recognizer {recognizer.Name}")
            break

        except (KeyError, ValueError, IndexError):
            opr.print_from("OPR Recognizer", "Recognizer not found!")
            continue


    while True:
        opr.list_choices(detector.get_supported_languages(as_dict=True).keys(), "Select a language")

        index_language = opr.input_from("OPR DetectRecog", "Select a language", 1)
        
        try:
            language = list(detector.get_supported_languages(as_dict=True).keys())[int(index_language) -1] 
            

            opr.print_from("OPR DetectRecog", f"Selected language {language}")
            break

        except (KeyError, ValueError, IndexError):
            opr.input_from("OPR DetectRecog", "Language not found!")
            continue

    detector_initialization_results = detector.initialize(language)

    if not detector_initialization_results:
        opr.print_from("OPR DetectRecog", "Detector initialization failed, defaulting...")
        detector.initialize(language=None)
    
    recognizer_initialization_results = recognizer.initialize(language)

    if not recognizer_initialization_results:
        opr.print_from("OPR DetectRecog", "Recognizer initialization failed, defaulting...")
        recognizer.initialize(language=None)

    opr.print_from("OPR DetectRecog", "Fully initialized! (Ctrl + c to exit)")
    
    while True:
        try:
            img_path = opr.input_from("OPR DetectRecog", "Enter the path to the image")
            img_spath = opr.clean_path(img_path)

            if not os.path.exists(img_spath):
                opr.print_from("OPR DetectRecog", "Image path is invalid, please try again!")
                continue

            img = Image.open(img_spath).convert('RGBA')
        
            detection_results = detector.detect_and_crop(img)
            recognition_results = []
            

            for bbox, _, image in detection_results:
                result = recognizer.recognize(image, bbox)
                for r in result:
                    recognition_results.append(r)

            opr.print_from("OPR DetectRecog", f"Finished!")
            for result in recognition_results:
                opr.print_from("OPR DetectRecog", f"{result.Text} at {result.QuadBox} with {result.Confidence} Confidence")

        except KeyboardInterrupt:
            opr.print_from("OPR DetectRecog", "Goodbye!", 2)
            break