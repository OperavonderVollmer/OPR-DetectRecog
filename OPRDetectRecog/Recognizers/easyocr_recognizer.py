from OperaPowerRelay import opr
from OPRDetectRecog.Interfaces.Recognizer import Recognizer
from OPRDetectRecog.Custom.LinguistResult import LinguistResult
import numpy

class easyocr_recognizer(Recognizer):

    def __init__(self, language=None, tolerance = None, path = None):
        super().__init__(language, tolerance, path)
        
        
        self._name = 'easyocr_recognizer'
        

    def get_supported_languages(self, as_dict = False) -> list[str] | dict[str, str]:
        easyocr_langs = {
            "english": "en",
            "japanese": "ja",
            "korean": "ko",
            "chinese_simplified": "ch_sim",
            "chinese_traditional": "ch_tra",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "italian": "it",
            "russian": "ru",
            "arabic": "ar",
            "thai": "th",
            "vietnamese": "vi",
            "indonesian": "id",
            "turkish": "tr",
            "persian": "fa",
            "hindi": "hi",
            "abaza": "abq",
            "adyghe": "ady",
            "afrikaans": "af",
            "angika": "ang",
            "assamese": "as",
            "avar": "ava",
            "azerbaijani": "az",
            "belarusian": "be",
            "bulgarian": "bg",
            "bihari": "bh",
            "bhojpuri": "bho",
            "bengali": "bn",
            "bosnian": "bs",
            "chechen": "che",
            "czech": "cs",
            "welsh": "cy",
            "danish": "da",
            "estonian": "et",
            "finnish": "fi",
            "irish": "ga",
            "goan_konkani": "gom",
            "croatian": "hr",
            "hungarian": "hu",
            "ingush": "inh",
            "icelandic": "is",
            "kabardian": "kbd",
            "kannada": "kn",
            "kurdish": "ku",
            "latin": "la",
            "lak": "lbe",
            "lezghian": "lez",
            "lithuanian": "lt",
            "latvian": "lv",
            "magahi": "mah",
            "maithili": "mai",
            "maori": "mi",
            "mongolian": "mn",
            "marathi": "mr",
            "malay": "ms",
            "maltese": "mt",
            "nepali": "ne",
            "newari": "new",
            "dutch": "nl",
            "norwegian": "no",
            "occitan": "oc",
            "pali": "pi",
            "polish": "pl",
            "portuguese": "pt",
            "romanian": "ro",
            "serbian_cyrillic": "rs_cyrillic",
            "serbian_latin": "rs_latin",
            "nagpuri": "sck",
            "slovak": "sk",
            "slovenian": "sl",
            "albanian": "sq",
            "swedish": "sv",
            "swahili": "sw",
            "tamil": "ta",
            "tabassaran": "tab",
            "telugu": "te",
            "tajik": "tjk",
            "tagalog": "tl",
            "uyghur": "ug",
            "ukrainian": "uk",
            "urdu": "ur",
            "uzbek": "uz",
        }
    
        if as_dict:
            return easyocr_langs
        else:
            return list(easyocr_langs.keys())

    def initialize(self, language, tolerance = None, path = None) -> bool:
        try:
            super().initialize(language, tolerance, path)

            import easyocr
            self._recognizor = easyocr.Reader([self._language], recog_network="standard")
        except KeyError:
            return False
        return True

    def recognize(self, frame, bbox) -> list[LinguistResult]:


        fr = numpy.array(frame)
        results = self._recognizor.readtext(fr)

        recognition_results = []
        if len(results) == 0: return recognition_results

        _, text, confidence = results[0]
            
        result = LinguistResult(bbox, text.strip(), confidence)

        recognition_results.append(result)



        return recognition_results
    

def get_recognizer() -> Recognizer:
    return easyocr_recognizer()