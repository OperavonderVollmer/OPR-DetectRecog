from abc import ABC, abstractmethod
import cv2
import numpy 
from OPRDetectRecog.Custom.Quadbox import QuadBox
from PIL import Image


class Detector(ABC):
    """
    Detector Interface

    This abstract class defines the interface for various detectors like EasyOCR, PaddleOCR, etc.

    Parameters
    ----------
    language : str
        The language to use for recognition.

    Attributes
    ----------
    language : str
        The language to use for recognition.
    name : str
        The name of the detector.
    detector : object
        The detector object.

    Methods
    -------
    detect(frame: numpy.ndarray) -> list[QuadBox]
        Detects objects or text within a given frame and returns a list of detected results.
    detect_and_crop(frame: numpy.ndarray) -> list[tuple[QuadBox, numpy.ndarray, Image.Image]]
        Detects objects or text within a given frame, crops the detected regions, and returns a list of cropped results.

    """
    def __init__(self, language: str = None, path: str = None):
        self._language = language or None
        self._path = path or None
        self._name = None
        self._detector = None

    @abstractmethod
    def get_supported_languages(self, as_dict: bool = False) -> list[str] | dict[str, str]:
        pass
    
    def initialize(self, language: str=None, path: str = None) -> bool:
        if not language:
            language = "chinese_simplified"    
        self._language = self.get_supported_languages(as_dict=True)[language]
        self._path = path


    @abstractmethod
    def detect(self, frame: numpy.ndarray) -> list[QuadBox]:
        """
        Detects objects or text within a given frame and returns a list of detected results.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame to analyze for detection.

        Returns
        -------
        list[QuadBox]
            A list of QuadBox instances containing bounding box information.
        """

        pass

    @abstractmethod
    def detect_and_crop(self, frame: numpy.ndarray) -> list[tuple[QuadBox, numpy.ndarray, Image.Image]]:
        """
        Detects objects or text within a given frame, crops the detected regions, and returns a list of cropped results.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame to analyze for detection and cropping.

        Returns
        -------
        list[tuple[QuadBox, numpy.ndarray, Image.Image]]
            A list of tuples containing a QuadBox instance with bounding box information, a numpy array of the cropped image, 
            and a PIL Image object of the cropped image.
        """
        pass

    def crop_rotated_box(self, image: numpy.ndarray, box: list[list[float]]) -> tuple[numpy.ndarray, Image.Image]:
        """
        Crops a rotated box from an image.

        Parameters
        ----------
        image : numpy.ndarray
            The image to crop from.
        box : list[list[float]]
            The 4 points of the box in the order of top-left, top-right, bottom-right, bottom-left.

        Returns
        -------
        tuple[numpy.ndarray, Image.Image]
            A tuple containing the cropped numpy array and a PIL Image object.
        """
        pts = numpy.array(box).astype(numpy.float32)

        width = int(max(numpy.linalg.norm(pts[0] - pts[1]), numpy.linalg.norm(pts[2] - pts[3])))
        height = int(max(numpy.linalg.norm(pts[0] - pts[3]), numpy.linalg.norm(pts[1] - pts[2])))

        dst = numpy.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=numpy.float32)
        M = cv2.getPerspectiveTransform(pts, dst)

        warped = cv2.warpPerspective(image, M, (width, height))

        pil_image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)).convert("RGB")
        
        return warped, pil_image
    
    @property
    def Name(self) -> str:
        return self._name