from dataset import *
from gesture import *

######### Gesture #########
# Creates leap motion unica dataset (Normalise and Sample)
def leapmotion_dataset(baseDir):
    # Crea file csv gesture
    # Rectangle
    ToolsDataset.create_gesture_dataset(baseDir, 'rectangle', 72)
    # Triangle
    ToolsDataset.create_gesture_dataset(baseDir, 'triangle', 54)
    # Caret
    ToolsDataset.create_gesture_dataset(baseDir, 'caret', 36)
    # V
    ToolsDataset.create_gesture_dataset(baseDir, 'v', 36)
    # X
    ToolsDataset.create_gesture_dataset(baseDir, 'x', 54)
    # Square Bracket Left
    ToolsDataset.create_gesture_dataset(baseDir, 'square-braket-left', 54)
    # Square Bracket Right
    ToolsDataset.create_gesture_dataset(baseDir, 'square-braket-right', 54)
    # Delete
    ToolsDataset.create_gesture_dataset(baseDir, 'delete', 54)
    #  Star
    ToolsDataset.create_gesture_dataset(baseDir, 'star', 90)
    # Left
    ToolsDataset.create_gesture_dataset(baseDir, 'left', 20)
    # Right
    ToolsDataset.create_gesture_dataset(baseDir, 'right', 20)

    return

# Makes one dollar dataset (Normalise and Sample)
def onedollar_dataset(baseDir):
    # Arrow
    ToolsDataset.create_gesture_dataset(baseDir, 'arrow/', 72)
    # Caret
    ToolsDataset.create_gesture_dataset(baseDir, 'caret/', 36)
    # Check
    ToolsDataset.create_gesture_dataset(baseDir, 'check/', 36)
    # Circle
    ToolsDataset.create_gesture_dataset(baseDir, 'circle/', 72)
    # Delete
    ToolsDataset.create_gesture_dataset(baseDir, 'delete_mark/', 54)
    # Left curly brace
    ToolsDataset.create_gesture_dataset(baseDir, 'left_curly_brace/', 36)
    # Right curly brace
    ToolsDataset.create_gesture_dataset(baseDir, 'right_curly_brace/', 36)
    # Left square bracket
    ToolsDataset.create_gesture_dataset(baseDir, 'left_sq_bracket/', 54)
    # Right square bracket
    ToolsDataset.create_gesture_dataset(baseDir, 'right_sq_bracket/', 54)
    # Pigtail
    ToolsDataset.create_gesture_dataset(baseDir, 'pigtail/', 36)
    # Question mark
    ToolsDataset.create_gesture_dataset(baseDir, 'question_mark/', 90)
    # Rectangle
    ToolsDataset.create_gesture_dataset(baseDir, 'rectangle/', 72)
    # Star
    ToolsDataset.create_gesture_dataset(baseDir, 'star/', 90)
    # Triangle
    ToolsDataset.create_gesture_dataset(baseDir, 'triangle/', 54)
    # V
    ToolsDataset.create_gesture_dataset(baseDir, 'v/', 36)
    # X
    ToolsDataset.create_gesture_dataset(baseDir, 'x/', 54)

######### Primitive #########
# Makes primitive dataset (from right and left movements)
def primitive_dataset(datasetDir, baseDir):
    # Up
    make_primitive_dataset(datasetDir, baseDir, Primitive.up)
    # Down
    make_primitive_dataset(datasetDir, baseDir, Primitive.down)
    # Diagonal
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -145) # - 145
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -135) # - 135
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -130) # - 130
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -60) # - 60
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -45) # - 45
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, 135) # 135
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, 60) # 60
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, 45) # 45
    # Forward
    make_primitive_dataset(datasetDir, baseDir, Primitive.forward)
    # Behind
    make_primitive_dataset(datasetDir, baseDir, Primitive.behind)

    return

datasetDir = '/home/alessandro/Scaricati/gestures/'
baseDir  = '/home/alessandro/Scaricati/dataset/leap_motion_unica/'

# Get all folders
folders = ToolsDataset.get_subdirectories(datasetDir)
folders = sorted(folders, key=lambda x: (str(re.sub('\D', '', x)), x))# Orders folders

# For each folder
for folder in folders:
    # Get files
    files = os.listdir(datasetDir + folder +'/')
    files = sorted(files, key=lambda x: (str(re.sub('\D', '', x)), x))# Orders files
    for file in files:

        # Index file and filename
        items = file.split('.')

        # Cartella per la gesture
        if not os.path.exists('/home/alessandro/Scaricati/csv/' + items[0]):
            os.makedirs('/home/alessandro/Scaricati/csv/' + items[0])

        # Copia contenuto
        copyfile(datasetDir+folder+'/'+file,
                 '/home/alessandro/Scaricati/csv/'+items[0]+'/'+folder+'_'+items[0]+'.csv')

for folder in '/home/alessandro/Scaricati/csv/':
    ToolsDataset.replace_csv('/home/alessandro/Scaricati/csv')


#primitive_dataset(datasetDir, baseDir)