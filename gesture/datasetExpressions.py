from enum import Enum
from model.gestureModel import Point, Line, Arc
# config
from config import Config
# CsvDataset
from dataset import CsvDatasetExtended

class TypeDataset(Enum):
    unistroke_1dollar = 0
    multistroke_1dollar = 1
    unica = 2
    shrec = 3

class DatasetFolders:

    @classmethod
    def returnFolders(cls, selected_dataset):
        if selected_dataset == TypeDataset.unistroke_1dollar:
            return cls.__returnDollar1Unistroke()
        if selected_dataset == TypeDataset.multistroke_1dollar:
            return cls.__returnDollar1Multistroke()
        if selected_dataset == TypeDataset.unica:
            return cls.__returnUnica()
        if selected_dataset == TypeDataset.shrec:
            return cls.__returnShrec()

    @staticmethod
    def __returnDollar1Unistroke():
        dataset_folder = "deictic/1dollar-dataset/resampled/"
        return {
            'arrow':            [CsvDatasetExtended(Config.baseDir +dataset_folder+ "arrow/")],
            'caret':            [CsvDatasetExtended(Config.baseDir +dataset_folder+ "caret/")],
            'check':            [CsvDatasetExtended(Config.baseDir +dataset_folder+ "check/")],
            'circle':           [CsvDatasetExtended(Config.baseDir +dataset_folder+ "circle/")],
            'delete_mark':      [CsvDatasetExtended(Config.baseDir +dataset_folder+ "delete_mark/")],
            'left_curly_brace': [CsvDatasetExtended(Config.baseDir +dataset_folder+ "left_curly_brace/")],
            'left_sq_bracket':  [CsvDatasetExtended(Config.baseDir +dataset_folder+ "left_sq_bracket/")],
            'pigtail':          [CsvDatasetExtended(Config.baseDir +dataset_folder+ "pigtail/")],
            'question_mark':    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "question_mark/")],
            'rectangle':        [CsvDatasetExtended(Config.baseDir +dataset_folder+ "rectangle/")],
            'right_curly_brace':[CsvDatasetExtended(Config.baseDir +dataset_folder+ "right_curly_brace/")],
            'right_sq_bracket': [CsvDatasetExtended(Config.baseDir +dataset_folder+ "right_sq_bracket/")],
            'star':             [CsvDatasetExtended(Config.baseDir +dataset_folder+ "star/")],
            'triangle':         [CsvDatasetExtended(Config.baseDir +dataset_folder+ "triangle/")],
            'v':                [CsvDatasetExtended(Config.baseDir +dataset_folder+ "v/")],
            'x':                [CsvDatasetExtended(Config.baseDir +dataset_folder+ "x/")],
        }

    @staticmethod
    def __returnDollar1Multistroke():
        dataset_folder = "deictic/mdollar-dataset/resampled/"
        return {
            'arrowhead' :           [CsvDatasetExtended(Config.baseDir +dataset_folder+ "arrowhead/")],
            'asterisk':             [CsvDatasetExtended(Config.baseDir +dataset_folder+ "asterisk/")],
            'D':                    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "D/")],
            'exclamation_point':    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "exclamation_point/")],
            'H':                    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "H/")],
            'half_note':            [CsvDatasetExtended(Config.baseDir +dataset_folder+ "half_note/")],
            'I':                    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "I/")],
            'N':                    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "N/")],
            'null':                 [CsvDatasetExtended(Config.baseDir +dataset_folder+ "null/")],
            'P':                    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "P/")],
            'pitchfork':            [CsvDatasetExtended(Config.baseDir +dataset_folder+ "pitchfork/")],
            'six_point_star':       [CsvDatasetExtended(Config.baseDir +dataset_folder+ "six_point_star/")],
            'T':                    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "T/")],
            'X':                    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "X/")],
        }

    @staticmethod
    def __returnUnica():
        pass

    @staticmethod
    def __returnShrec():
        dataset_folder = "deictic/shrec-dataset/resampled/index_tip/"
        return {
            'gesture_2':    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_2/")],
            'gesture_7':    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_7/")],
            'gesture_8':    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_8/")],
            'gesture_9':    [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_9/")],
            'gesture_10':   [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_10/")],
            'gesture_11':   [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_11/")],
            'gesture_12':   [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_12/")],
            'gesture_13':   [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_13/")],
            'gesture_14':   [CsvDatasetExtended(Config.baseDir +dataset_folder+ "gesture_14/")],
        }

class DatasetExpressions:

    @staticmethod
    def returnExpressions(selected_dataset):
        if selected_dataset == TypeDataset.unistroke_1dollar:
            return DatasetExpressions.__returnDollar1Unistroke()
        if selected_dataset == TypeDataset.multistroke_1dollar:
            return DatasetExpressions.__returnDollar1Multistroke()
        if selected_dataset == TypeDataset.unica:
            return DatasetExpressions.__returnUnica()
        if selected_dataset == TypeDataset.shrec:
            return DatasetExpressions.__returnShrec()

    @staticmethod
    def __returnDollar1Unistroke():
        return {
            'arrow': [Point(0, 0) + Line(6, 4) + Line(-4, 0) + Line(5, 1) + Line(-1, -4)],
            'caret': [Point(0, 0) + Line(2, 3) + Line(2, -3)],
            'check': [Point(0, 0) + Line(2, -2) + Line(4, 6)],
            'circle': [Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False)],
            'delete_mark': [Point(0, 0) + Line(2, -3) + Line(-2, 0) + Line(2, 3)],
            'left_curly_brace': [Point(0, 0) + Arc(-5, -5, cw=False) + Arc(-3, -3) + Arc(3, -3) + Arc(5, -5, cw=False)],
            'left_sq_bracket': [Point(0, 0) + Line(-4, 0) + Line(0, -5) + Line(4, 0)],
            'pigtail': [Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False)],
            'question_mark': [Point(0,0) + Arc(4,4) + Arc(4,-4) + Arc(-4,-4) + Arc(-2,-2, cw=False) + Arc(2, -2, cw=False)],
            'rectangle': [Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0)],
            'right_curly_brace': [Point(0,0) + Arc(5,-5) +  Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Arc(-5,-5)],
            'right_sq_bracket': [Point(0,0) + Line(4,0) + Line(0, -5) + Line(-4, 0)],
            'star': [Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3)],
            'triangle': [Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)],
            'v': [Point(0,0) + Line(2,-3) + Line(2,3)],
            'x': [Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3)]
        }

    @staticmethod
    def __returnDollar1Multistroke():
        return {
            'arrowhead': [Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2),
                          Point(4, 2) + Line(2, -2) + Line(-2, -2) + Point(0, 0) + Line(6, 0)],
            'asterisk': [Point(4, 4) + Line(-4, -4) + Point(0, 4) + Line(4, -4) + Point(2, 4) + Line(0, -4)],
            'D': [#Point(0, 0) + Line(0, 6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0),
                  Point(0, 6) + Line(0, -6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3,cw=True) + Line(-2, 0),
                  #Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 6),
                  #Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 6) + Line(0, -6)]),
            ],
            'exclamation_point': [Point(0, 4) + Line(0, -3) + Point(0, 1) + Line(0, -1)],
            'H': [Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0) + Point(4, 4) + Line(0, -4),
                  Point(0, 4) + Line(0, -4) + Point(4, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0),
                  #Point(4, 4) + Line(0, -4) + Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0)]),
                 ],
            'half_note': [Point(0,0)+Arc(-3,3,cw=False)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False) + Arc(3,3,cw=False)+Point(2,16)+Line(0,-20),
                          Point(2,16)+Line(0,-20)+Point(0, 0)+Arc(-3,3,cw=False)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False)+Arc(3,3,cw=False),
                          Point(0,0)+Arc(-3,3,cw=False)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False)+Arc(3,3,cw=False)+Point(2,-4)+Line(0, 20),
                          Point(2,-4)+Line(0,20)+Point(0,0)+Arc(-3,3,cw=False)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False)+Arc(3,3,cw=False)],
            'I': [Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0),
                  Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0) + Point(0, 4) + Line(4, 0),
                  Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0) + Point(2, 0) + Line(0, 4),
                  Point(2, 4) + Line(0, -4) + Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0)],
            'N': [Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4)],
            'null': [Point(0,0)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False)+Arc(3,3,cw=False)+Arc(-3,3,cw=False)+Point(4,1)+Line(-8,-8),  # 410
                     Point(0,0)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False)+Arc(3,3,cw=False)+Arc(-3,3,cw=False)+Point(-4,-7)+Line(8,8)],  # 118
            'P': [#Point(0, 0) + Line(0, 8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
                  Point(0, 8) + Line(0, -8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
                  #Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 8) + Line(0, -8),
                  #Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 8)]),
            ],
            'pitchfork': [Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False) + Point(0, 4) + Line(0, -4),
                          Point(0, 4) + Line(0, -4) + Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False)],
            'six_point_star': [Point(0,0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) + Point(-2, -1) + Line(4, 0) + Line(-2, -4) + Line(-2,4),
                               Point(-2,-1) + Line(4, 0) + Line(-2, -4) + Line(-2, 4) + Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2,4),
                               Point(-2,-2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(4, 0) + Line(-2, -4) + Line(-2,4),
                               Point(-2,1) + Line(4, 0) + Line(-2, -4) + Line(-2, 4) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0),
                               Point(-2,-2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(2, -4) + Line(2, 4) + Line(-4, 0),
                               Point(-2,1) + Line(2, -4) + Line(2, 4) + Line(-4, 0) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0),
                               Point(0,0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) + Point(-2, -1) + Line(2, -4) + Line(2, 4) + Line(-4, 0),
                               Point(-2,-1) + Line(2, -4) + Line(2, 4) + Line(-4, 0) + Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4)],
            'T': [Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4),
                  #Point(-2, 0) + Line(4, 0) + Point(-4, 2) + Line(0, 4),
                  #Point(2, 0) + Line(-4, 0) + Point(0, 0) + Line(0, -4),
                  #Point(2, 0) + Line(-4, 0) + Point(-4, 2) + Line(0, 4),
                  Point(0, 0) + Line(0, -4) + Point(-2, 0) + Line(4, 0),
                  #Point(0, 0) + Line(0, -4) + Point(2, 0) + Line(-4, 0),
                  #Point(-4, 2) + Line(0, 4) + Point(-2, 0) + Line(4, 0),
                  #Point(-4, 2) + Line(0, 4) + Point(2, 0) + Line(-4, 0)
            ],
            'X': [#Point(0, 0) + Line(4, 4) + Point(4, 0) + Line(-4, 4), NO
                  Point(0, 0) + Line(4, 4) + Point(0, 4) + Line(4, -4),  # 33
                  #Point(4, 4) + Line(-4, -4) + Point(4, 0) + Line(-4, 4), NO
                  Point(4, 4) + Line(-4, -4) + Point(0, 4) + Line(4, -4),  # 174
                  #Point(4, 0) + Line(-4, 4) + Point(0, 0) + Line(4, 4), # NO
                  #Point(4, 0) + Line(-4, 4) + Point(4, 4) + Line(-4, -4), #NO
                  Point(0, 4) + Line(4, -4) + Point(0, 0) + Line(4, 4), # 44
                  Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(-4, -4)], # 368
        }

    @staticmethod
    def __returnUnica():
        return None

    @staticmethod
    def __returnShrec():
        return {

            'gesture_2': [
                Point(0, 0) + Line(-4, 2),
                Point(0, 0) + Line(-4, 2) + Line(4, -2),
                #Point(0, 0) + Line(0, -4) + Line(-4, 2),
                #Point(0, 0) + Line(2, -4) + Line(2, 4),
            ],

            'gesture_7': [
                Point(0, 0) + Line(4, 0),
                Point(0,0) + Line(4,0) + Line(-4,0)
            ],

            'gesture_8': [
                Point(4, 0) + Line(-4, 0),
                Point(4, 0) + Line(-4, 0) + Line(4, 0)
            ],

            'gesture_9': [
                Point(0, 0) + Line(0, 4),
                Point(0, 0) + Line(0, 4) + Line(0, -4)
            ],

            'gesture_10': [
                Point(0, 4) + Line(0, -4),
                Point(0, 4) + Line(0, -4) + Line(0, 4)
            ],

            'gesture_11': [
                Point(0, 4) + Line(4, -4) + Line(-4, 0) + Line(4,4),
                Point(4, 4) + Line(-4, -4) + Line(0, 4) + Line(4,-4)
            ],

            'gesture_12': [
                Point(0,4) + Line(0,-4) + Line(-2,2) + Line(4,0),
                Point(0,2) + Line(0,2) + Line(0,-4) + Line(-2,2) + Line(4,0),
                ],

            'gesture_13': [
                Point(2, 4) + Line(-2, -4) + Line(-2, 4),
                Point(-2, 4) + Line(2, -4) + Line(2, 4)
            ]
        }
