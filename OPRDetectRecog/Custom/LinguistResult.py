from OPRDetectRecog.Custom.Quadbox import QuadBox



class LinguistResult:
    def __init__(self, QuadBox: QuadBox|list, original: str, translated: str, confidence: float):
        self._QuadBox = self.process_bbox(QuadBox)
        self._original = original
        self._translated = translated
        self._confidence = confidence

    def process_bbox(self, bbox: QuadBox|list) -> QuadBox:
        
        if isinstance(bbox, QuadBox):
            return bbox
        else:
            return QuadBox(bbox)


    @property
    def QuadBox(self) -> QuadBox:
        return self._QuadBox
    
    @property
    def Original(self) -> str:
        return self._original

    @property
    def Translated(self) -> str:
        return self._translated
    

    @property
    def Confidence(self) -> float:
        return self._confidence