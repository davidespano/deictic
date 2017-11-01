from test import *
from os import *

# Class Deictic
class Deictic():
    def __init__(self, type_gestures, base_dir, n_states = 6, n_sample = 20):
        # Type (unistroke or multistroke)
        self.type_gestures = type_gestures
        # Num states and num samples
        # Gesture dataset
        if(self.type_gestures == "unistroke"):
            # Gestures
            self.gestures = \
            [
                "arrow", "caret",
                "circle", "check",
                "delete_mark", "left_curly_brace",
                "left_sq_bracket", "pigtail",
                "question_mark", "rectangle",
                "right_curly_brace", "right_sq_bracket",
                "star", "triangle",
                "v", "x"
            ]
            # Gestures and the respective number of samples (used for creating the hmm and during the comparison)
            self.gestures_num_samples = \
            [
                ("arrow", 4 * n_sample), ("caret", 2 * n_sample),
                ("circle", 4*n_sample), ("check", 2*n_sample),
                ("delete_mark", 3*n_sample), ("left_curly_brace", 6*n_sample),
                ("left_sq_bracket", 3*n_sample), ("pigtail", 4*n_sample),
                ("question_mark", 4*n_sample), ("rectangle", 4*n_sample),
                ("right_curly_brace", 6*n_sample), ("right_sq_bracket", 3*n_sample),
                ("star", 5*n_sample), ("triangle", 3*n_sample),
                ("v", 2*n_sample), ("x", 3*n_sample)
            ]

        # HMMs
        self.hmms = []
        # Creates models
        Parse.setStatesSamples(n_states, n_sample)
        for gesture in self.gestures:
            # Gets the hmm
            model = Parse.parseExpression(self.type_gestures + "-" + gesture)
            self.hmms.append(model)
        # Base dir
        self.base_dir = base_dir


    # Tests models given dataset
    def test(self):
        # Normalizes data and gets data
        sequences = self.__normalizeAndGetsData()
        # Creates test class object
        test_models = test(self.hmms, self.gestures)
        # Starts test
        return test_models.single_file(sequences)


    # Applies normalize transforms
    def __normalizeAndGetsData(self):
        # Delete files from normalize folder
        filelist = [f for f in os.listdir(".") if f.endswith(".csv")]
        for f in filelist:
            os.remove(f)

        # Creates a sequence for each gesture, according to the number of its components
        sequences = []

        # Dataset
        dataset = CsvDataset(self.base_dir)

        for gesture in self.gestures_num_samples:
            # Transform
            transform1 = NormaliseLengthTransform(axisMode=True)
            transform2 = ScaleDatasetTransform(scale=100)
            transform3 = CenteringTransform()
            if self.type_gestures == "unistroke":
                transform5 = ResampleInSpaceTransform(gesture[1])
            else:
                transform5 = \
                    ResampleInSpaceTransformMultiStroke(self.n_sample,
                                                        strokes=self.gestures[2])
            # Apply transforms
            dataset.addTransform(transform1)
            dataset.addTransform(transform2)
            dataset.addTransform(transform3)
            dataset.addTransform(transform5)
            # Saves sequence
            dataset.applyTransforms(self.base_dir+'normalize_'+ gesture[0] +'/')
            # Adds the last computed sequence in the list
            sequences.append(CsvDataset(self.base_dir+'normalize_'+ gesture[0] +'/').readFile("gesture.csv"))

        return sequences



    # Gets gesture name
    def __getsGestureName(self):
        # Gets file in dataset
        for file in os.listdir(self.dir):
            # Gets number of samples
            name = (file.split('\\')[0]).split('.')[0]
        return name
