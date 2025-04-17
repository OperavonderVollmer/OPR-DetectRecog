from OPRDetectRecog.Custom.Quadbox import QuadBox



class LinguistResult:
    def __init__(self, QuadBox: QuadBox|list, text: str, confidence: float):
        self._QuadBox = self.process_bbox(QuadBox)
        self._text = text
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
    def Text(self) -> str:
        return self._text
    

    @property
    def Confidence(self) -> float:
        return self._confidence