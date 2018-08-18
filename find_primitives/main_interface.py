from dataset import CsvDatasetExtended
from interface import Window
from config import Config

gesture_datasets = {
     'arrow': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/arrow/")]),
     'caret': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/caret/")]),
     'check': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/check/")]),
     'circle':(4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/circle/")]),
     'delete_mark': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/delete_mark/")]),
    # 'left_curly_brace': (6,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/left_curly_brace/")]),
     'left_sq_bracket': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/left_sq_bracket/")]),
    # 'pigtail': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/pigtail/")]),
    # 'question_mark': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/question_mark/")]),
     'rectangle': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/rectangle/")]),
    # 'right_curly_brace': (6,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/right_curly_brace/")]),
     'right_sq_bracket': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/right_sq_bracket/")]),
     'star': (5,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/star/")]),
     'triangle': (3, [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/triangle/")]),
     'v': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/v/")]),
     #'x': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/x/")])
}
window = Window(datasets=gesture_datasets)

