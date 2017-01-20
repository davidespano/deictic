from dataset import *
from pomegranate import *
from topology import *
import matplotlib.pyplot as plt
import random
from gesture import *
from test import *
from enum import Enum
import xml.etree.ElementTree as ET
import csv
import xmlutils.xml2csv


def xml_to_csv(path_read, path_write):
    converter = xmlutils.xml2csv(path_read, path_write, encoding="utf-8")
    converter.convert(tag="item")

    return

def create_original_csv(path, baseDir):
    index = 0
    # Prendi tutte le cartelle
    folders = LeapDataset.get_immediate_subdirectories(path)
    folders = sorted(folders, key=lambda x: (int(re.sub('\D', '', x)), x))
    # Per ogni cartella
    for folder in folders:
        # Per ogni tipologia (slow, medium, fast)
        subfolders = LeapDataset.get_immediate_subdirectories(path+folder)
        for subfolder in subfolders:
            for file in os.listdir(path + folder + subfolder):
                gesture_name = file.split('_')[0] # Nome Gesture
                # Cartella per la gesture
                if not os.path.exists(baseDir + 'original/' + subfolder + '/' +  gesture_name):
                    os.makedirs(baseDir + 'original/' + subfolder)
                # Copia contenuto
                index = index + 1
                xml_to_csv(path + folder + '/' + subfolder + '/' + file, baseDir + 'original' + subfolder + '/' + gesture_name + '/' + gesture_name + '{}.csv'.format(index))
    return


baseDir = '/home/alessandro/Scaricati/csv/'
path = '/home/alesssandro/Scaricati/xml_logs/'