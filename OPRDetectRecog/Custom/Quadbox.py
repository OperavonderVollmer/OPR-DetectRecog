import math
from OperaPowerRelay import opr
import numpy




class QuadBox:
    """
    A QuadBox is a representation of a quadrilateral in 2D space, with attributes of the box's points and angle of rotation.

    Parameters
    ----------
    points : list[tuple[float, float]]
        A list of the four points making up the box, in order of top-left, top-right, bottom-right, bottom-left.
    angle : float | None, optional
        The angle of rotation of the box, by default None which will be calculated from the points.
    padding : int, optional
        The amount of padding to be added to the box, by default 5.
        
    Attributes
    ----------
    points : list[tuple[float, float]]
        A list of the four points making up the box, in order of top-left, top-right, bottom-right, bottom-left.
    angle : float | None
        The angle of rotation of the box, by default None which will be calculated from the points.
    padding : int | 5
        The amount of padding added to the box.

        
    Properties
    ----------
    Area : float
        The area of the box.
    AreaStrict : float
        The area of the box, calculated from the exact points.
    Width : float
        The width of the box.
    Height : float  
        The height of the box.
    Center : tuple[float, float]
        The center point of the box.
    CenterStrict : tuple[float, float]
        The center point of the box, calculated from the exact points.
    Top : tuple[float, float]
        The top point of the box, calculated from the top-left and top-right points.
    TopLeft : tuple[float, float]
        The top-left point of the box.
    TopRight : tuple[float, float]
        The top-right point of the box.
    Bottom : tuple[float, float]
        The bottom point of the box, calculated from the bottom-left and bottom-right points.
    BottomLeft : tuple[float, float]
        The bottom-left point of the box.
    BottomRight : tuple[float, float]
        The bottom-right point of the box.
    Left : tuple[float, float]
        The left point of the box, calculated from the top-left and bottom-left points.
    Right : tuple[float, float]
        The right point of the box, calculated from the top-right and bottom-right points.
    """
    
    def __init__(self, 
            points: list[tuple[float, float]],
            angle: float | None = None,
            padding: int = 5):

            if len(points) != 4:
                opr.error_pretty(ValueError, "OpheliaVisorR | QuadBox", f"Invalid points! {points}", "HUMAN ERROR")
                raise ValueError("QuadBox must have 4 points")        

            if padding > 0:
                self._points = [p for p in points]
                self._points[0] = (points[0][0] - padding, points[0][1] - padding)
                self._points[1] = (points[1][0] + padding, points[1][1] - padding)
                self._points[2] = (points[2][0] + padding, points[2][1] + padding)
                self._points[3] = (points[3][0] - padding, points[3][1] + padding)

                self._angle = self.calculate_angle() if angle is None else angle
                return
            
            self._points = points
            self._angle = self.calculate_angle() if angle is None else angle

    def calculate_angle(self) -> float:
        p0 = self.to_numpy[0]
        p1 = self.to_numpy[1]

        delta_x = p1[0] - p0[0]
        delta_y = p1[1] - p0[1]

        angle_rad = math.atan2(delta_y, delta_x)
        return math.degrees(angle_rad)


    @property
    def Points(self) -> list[tuple[float, float]]:
        return self._points
    
    @property
    def to_numpy(self) -> numpy.ndarray:
        return numpy.array(self._points, dtype=numpy.float32)
    
    @property
    def to_tuple(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], float]:
        
        return (self._points[0], self._points[1], self._points[2], self._points[3], self._angle)


    @property
    def Width(self) -> float:
        return float(numpy.linalg.norm(self.to_numpy[0] - self.to_numpy[1]))

    @property
    def Height(self) -> float:
        return float(numpy.linalg.norm(self.to_numpy[0] - self.to_numpy[3]))
    
    @property
    def TopLeft(self) -> tuple[float, float]:
        return self._points[0]

    @property
    def TopRight(self) -> tuple[float, float]:
        return self._points[1]
    
    @property
    def BottomRight(self) -> tuple[float, float]:
        return self._points[2]
    
    @property
    def BottomLeft(self) -> tuple[float, float]:
        return self._points[3]
    
    @property
    def Top(self) -> tuple[float, float]:
        x = (self.TopLeft[0] + self.TopRight[0]) / 2
        y = (self.TopLeft[1] + self.TopRight[1]) / 2
        return x, y

    @property
    def Bottom(self) -> tuple[float, float]:
        x = (self.BottomLeft[0] + self.BottomRight[0]) / 2
        y = (self.BottomLeft[1] + self.BottomRight[1]) / 2
        return x, y
    
    @property
    def Left(self) -> tuple[float, float]:
        x = (self.TopLeft[0] + self.BottomLeft[0]) / 2
        y = (self.TopLeft[1] + self.BottomLeft[1]) / 2
        return x, y
    
    @property
    def Right(self) -> tuple[float, float]:
        x = (self.TopRight[0] + self.BottomRight[0]) / 2
        y = (self.TopRight[1] + self.BottomRight[1]) / 2
        return x, y
    
    @property
    def Center(self) -> tuple[float, float]:
        x = (self.Top[0] + self.Bottom[0]) / 2
        y = (self.Top[1] + self.Bottom[1]) / 2
        return x, y
    
    @property
    def CenterStrict(self) -> tuple[float, float]:
        x = sum(p[0] for p in self._points) /4
        y = sum(p[1] for p in self._points) /4
        return x, y


    @property
    def Area(self) -> float:
         return self.Height * self.Width
    
    @property
    def AreaStrict(self) -> float:
        x = self.to_numpy[:, 0]
        y = self.to_numpy[:, 1]

        x = numpy.append(x, x[0])
        y = numpy.append(y, y[0])

        return 0.5 * numpy.abs(numpy.dot(x[:-1], y[1:]) - numpy.dot(x[1:], y[:-1]))

    @property
    def Angle(self) -> float:
        return self._angle



    def __repr__(self) -> str:
        return f"QuadBox: {self.to_tuple}"