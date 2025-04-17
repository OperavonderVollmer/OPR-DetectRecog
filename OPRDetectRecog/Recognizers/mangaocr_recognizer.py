from OperaPowerRelay import opr
from OPRDetectRecog.Interfaces.Recognizer import Recognizer
from OPRDetectRecog.Custom.LinguistResult import LinguistResult


class mangaocr_recognizer(Recognizer):

    def __init__(self, language=None, tolerance = None, path = None):
        super().__init__(language, tolerance, path)
        
        
        self._name = 'mangaocr_recognizer'

    def get_supported_languages(self, as_dict = False):
        mangaocr_langs = {'MangaOCR doesn''t use languages': 'MangaOCR doesn''t use languages'}

        if as_dict:
            return mangaocr_langs

        return list(mangaocr_langs.keys())
    


    def initialize(self, language, tolerance = None, path = None) -> bool:

        # manga ocr doesn't use language, tolerance, or a path

        from manga_ocr import MangaOcr
    
        self._recognizor = MangaOcr()

        return True

    def recognize(self, frame, bbox) -> list[LinguistResult]:

        result = self._recognizor(frame)

        recognition_results = [LinguistResult(bbox, result.strip(), -1.0)]
        

        return recognition_results
    

def get_recognizer() -> Recognizer:
    return mangaocr_recognizer()