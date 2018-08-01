from dataset import CsvDatasetExtended
from interface import Window
from config import Config

gesture_datasets = {
    'arrow': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/arrow/")],
    # 'caret': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/caret/")],
    # 'check': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/check/")],
    # 'circle':[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/circle/")],
    # 'delete_mark': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/delete_mark/")],
    # 'left_curly_brace': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/left_curly_brace/")],
    # 'left_sq_bracket': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/left_sq_bracket/")],
    # 'pigtail':[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/pigtail/")],
    # 'question_mark': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/question_mark/")],
    # 'rectangle': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/rectangle/")],
    # 'right_curly_brace':[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/right_curly_brace/")],
    # 'right_sq_bracket': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/right_sq_bracket/")],
    # 'star': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/star/")],
    # 'triangle': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/triangle/")],
    # 'v': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/v/")],
    # 'x': [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/raw/x/")]
}
window = Window(datasets=gesture_datasets)

