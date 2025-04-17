from OperaPowerRelay import opr
from abc import ABC, abstractmethod
from PIL import Image
from OPRDetectRecog.Custom.Quadbox import QuadBox
from OPRDetectRecog.Custom.LinguistResult import LinguistResult

class Recognizer(ABC):
    
    def __init__(self, language: str=None, tolerance: float=None, path: str=None):
        self._language = language or None
        self._tolerance = tolerance or None
        self._path = path or None
        self._name = None
        self._recognizer = None


    @abstractmethod
    def get_supported_languages(self, as_dict: bool = False) -> list[str] | dict[str, str]:
        pass

    def initialize(self, language: str = None, tolerance: float=None, path: str=None) -> bool:
        if not language:
            language = "chinese_simplified"    
        self._language = self.get_supported_languages(as_dict=True)[language] 
        self._tolerance = tolerance or None
        self._path = path or None


    @abstractmethod 
    def recognize(self, frame: Image.Image, bbox: QuadBox) -> list[LinguistResult]:
        """
        Recognizes text within a specified image frame and bounding box.

        This method processes the provided image frame to identify and extract 
        text within the specified bounding box. It returns a list of 
        LinguistResult instances, which contain the recognized text, the 
        associated bounding box, and confidence levels for each detection.

        Parameters
        ----------
        frame : Image.Image
            The image frame to process for text recognition.
        bbox : QuadBox
            The bounding box that defines the area of the image frame to 
            search for text.

        Returns
        -------
        list[LinguistResult]
            A list of LinguistResult objects, each comprising the recognized 
            text, the bounding box where the text was found, and the confidence 
            level of the recognition.
        """

        pass


    @property
    def Name(self) -> str:
        return self._name