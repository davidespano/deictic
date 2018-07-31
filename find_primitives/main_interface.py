from dataset import CsvDatasetExtended, Sequence
from interface import Window
from tkinter import *
from config import Config

gesture_datasets = {"arrow":[CsvDatasetExtended(dir=Config.baseDir+'deictic/1dollar-dataset/raw/arrow/')]}
window = Window(datasets=gesture_datasets)

