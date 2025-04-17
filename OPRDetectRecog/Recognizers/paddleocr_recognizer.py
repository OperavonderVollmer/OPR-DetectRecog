from OPRDetectRecog.Interfaces.Recognizer import Recognizer
from OPRDetectRecog.Custom.LinguistResult import LinguistResult
import numpy


class paddleocr_recognizer(Recognizer):

    def __init__(self, language=None, tolerance = None, path = None):
        super().__init__(language, tolerance, path)
        

        self._name = 'paddleocr_recognizer'
        
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

    def initialize(self, language, tolerance = None, path = None) -> bool:
        try:
            super().initialize(language, tolerance, path)
        
            from paddleocr import PaddleOCR 

            if path is not None:
                self._recognizor = PaddleOCR(use_gpu=True, rec_model_dir=path, show_log=False)
            else:

                """
                
                    Again, should replace these parameters with the ones that work the best on your machine
                
                """
                self._recognizor = PaddleOCR(use_gpu=True, show_log=False, providers = ['DmlExecutionProvider'])

        except KeyError:
            return False
        return True


    def recognize(self, frame, bbox) -> list[LinguistResult]:

        fr = numpy.array(frame)
        results = self._recognizor.ocr(fr, cls=True, rec=True, det=True)

        recognition_results = []
        
        print(results)
        if not results[0]: return recognition_results

        for i in results[0]:
            text, confidence = i[1]

            result = LinguistResult(bbox, text.strip(), confidence)

            recognition_results.append(result)


        return recognition_results
    

def get_recognizer() -> Recognizer:
    return paddleocr_recognizer()