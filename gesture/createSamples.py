from gesture import *
from test import *
from lxml import etree
import re

# Create the folder for origina, normalize and down-trajectory
def create_folder(baseDir, nome, operator = '', is_ground = True):
    if(is_ground == True):
        # Creazione cartelle
        if not os.path.exists(baseDir + 'original/' + nome):
            os.makedirs(baseDir + 'original/' + nome)
        if not os.path.exists(baseDir + 'normalised-trajectory/' + nome):
            os.makedirs(baseDir + 'normalised-trajectory/' + nome)
        if not os.path.exists(baseDir + 'down-trajectory/' + nome):
            os.makedirs(baseDir + 'down-trajectory/' + nome)
    else:
        # Creazione cartelle per gesture composte
        if not os.path.exists(baseDir + operator+'/'+nome):
            os.makedirs(baseDir + operator+'/'+nome)

def xml_to_csv(path_read, path_write):
    data = open('/home/alessandro/Scaricati/xml_logs/conversion.xslt')
    xslt_content = data.read()
    xslt_root = etree.XML(xslt_content)
    dom = etree.parse(path_read)
    transform = etree.XSLT(xslt_root)
    result = transform(dom)
    f = open(path_write, 'w')
    f.write(str(result))
    f.close()

    return

def create_original_files(path, baseDir):
    index_user = 0
    index_type = 0
    index_file = 0
    # Prendi tutte le cartelle
    folders = LeapDataset.get_immediate_subdirectories(path)
    folders = sorted(folders, key=lambda x: (int(re.sub('\D', '', x)), x))# Orders folders

    # Per ogni cartella
    for folder in folders:
        index_user = index_user + 1 # User Index
        # Per ogni tipologia (slow, medium, fast)
        subfolders = LeapDataset.get_immediate_subdirectories(path+'/'+folder)
        #subfolders = sorted(subfolders, key=lambda x: (int(re.sub('\D', '', x)), x))# Orders subfolders

        for subfolder in subfolders:
            index_type = index_type + 1 # Type Index
            files = os.listdir(path + folder +'/' + subfolder)
            files = sorted(files, key=lambda x: (int(re.sub('\D', '', x)), x))# Orders files

            for file in files:
                index_file = index_file + 1 # File Index

                items = re.findall('\d*\D+', file)

                # Cartella per la gesture
                if not os.path.exists(baseDir + 'original/' +  items[0]):
                    os.makedirs(baseDir + 'original/' + items[0])
                # Copia contenuto
                xml_to_csv(path + folder + '/' + subfolder + '/' + file, baseDir + 'original/' + items[0] + '/' + items[0]+'_'+subfolder + '_{}_{}_{}.csv'.format(index_user, index_type, index_file))

            # Azzeramento indici
            index_file = 0
        index_type = 0
    return

def create_gestures_dataset(baseDir, sample=20):
    # Get all folders
    folders = LeapDataset.get_immediate_subdirectories(baseDir+'original/')

    # Create Normalize and Down-Trajectory from original files
    for folder in folders:
        create_gesture_dataset(baseDir, folder+'/', sample=sample)
        print(folder + ' completed');

def create_gesture_dataset(baseDir, path, sample=20):
    # Creazione cartelle
    create_folder(baseDir, path)
    # Creazione dataset originale
    #create_original_files(baseDir, path_to_save)
    dataset = LeapDataset(baseDir + 'original/' + path)
    # Normalizzazione
    dataset.normalise(baseDir + 'normalised-trajectory/' + path, norm_axis=True)
    dataset = LeapDataset(baseDir + 'normalised-trajectory/' + path)
    # DownSample
    dataset.down_sample(baseDir + 'down-trajectory/' + path, sample)

    # Plot
    dataset = LeapDataset(baseDir + 'down-trajectory/' + path + '/')
    #for filename in dataset.getCsvDataset():
    #    sequence = dataset.read_file(filename, dimensions=2, scale=100)
    #    plt.axis("equal")
    #    plt.plot(sequence[:, 0], sequence[:, 1])
    #    plt.show()


baseDir = '/home/alessandro/Scaricati/dataset/$1/'
path_to_save = '/home/alessandro/Scaricati/xml_logs/'
# Original csv, Normalize and Down-Trajectory files
create_gestures_dataset(baseDir)