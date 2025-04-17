from OPRDetectRecog.Custom.Quadbox import QuadBox
from OPRDetectRecog.Interfaces.Detector import Detector
import numpy
import os


class paddleocr_detector(Detector):

    def __init__(self, language=None, path=None):
        super().__init__(language, path)
        
        self._name = "paddleocr_detector"
        
    def initialize(self, language=None, path = None) -> bool:
        try:
            super().initialize(language, path)
            
            from paddleocr import PaddleOCR


            """
            Notes! 

            You should probably play around with these parameters. These are just the ones that work the best on my machine. However, use_angle_cls is a must for detection
            """

            if path is not None and os.path.exists(path):
                self._detector = PaddleOCR(
                    show_log=False,
                    lang=self._language,
                    use_angle_cls=True, 
                    rec_model_dir=path, 
                    use_gpu = True
                )

            else:
                self._detector = PaddleOCR(
                    show_log=False,
                    lang=self._language,
                    use_angle_cls=True, 
                    providers = ['DmlExecutionProvider'], 
                    use_gpu = True
                )

        except KeyError:
            return False
        return True

    def get_supported_languages(self, as_dict: bool = False) -> list[str] | dict[str, str]:
        paddle_langs = {
            "english": "en",
            "chinese_simplified": "ch",           # default if omitted
            "chinese_traditional": "chinese_cht",
            "japanese": "japan",
            "korean": "korean",
            "french": "french",
            "german": "german",
            "spanish": "spanish",
            "portuguese": "portuguese",
            "italian": "italian",
            "russian": "russian",
            "arabic": "arabic",
            "turkish": "turkish",
            "thai": "thai",
            "hindi": "hindi",
            "vietnamese": "vietnamese",
            "persian": "persian",
            "mongolian": "mongolian",
            "kazakh": "kazakh",
            "tamil": "tamil",
            "telugu": "telugu",
            "marathi": "marathi",
            "bangla": "bangla",
            "urdu": "urdu",
            "romanian": "romanian",
            "ukrainian": "ukrainian",
            "greek": "greek",
            "cyrillic": "cyrillic",               # includes Russian, Bulgarian, etc.
            "serbian": "serbian",
            "kannada": "kannada",
            "malayalam": "malayalam",
            "lao": "lao",
            "burmese": "burmese",
            "khmer": "khmer",
            "nepali": "nepali",
            "sinhalese": "sinhalese",
            "sundanese": "sundanese",
            "javanese": "javanese",
            "hebrew": "hebrew",
            "macedonian": "macedonian"
        }

        if as_dict:
            return paddle_langs

        return list(paddle_langs.keys())
    
    def detect(self, frame: numpy.ndarray) -> list[QuadBox]:

        img = self._detector.ocr(frame, rec=False, cls=True, det=True)

        processed_results = []
        
        for i in img[0]:
            bbox = [i[0], i[1], i[2], i[3]]     

            result = QuadBox(bbox)

            processed_results.append(result)

        return processed_results
    

    def detect_and_crop(self, frame):
        fr = numpy.array(frame)
        img = self._detector.ocr(fr, rec=False, cls=True, det=True)

        cropped_results = []

        for i in img[0]:
            bbox = [i[0], i[1], i[2], i[3]]
            
            result = QuadBox(bbox)

            warped, pil_image = self.crop_rotated_box(fr, result.Points)


            cropped_results.append((result, warped, pil_image))

        return cropped_results
    
def get_detector() -> Detector:
    return paddleocr_detector()