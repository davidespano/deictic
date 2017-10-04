from test import *
from os import *

# Class Deictic
class Deictic():
    def __init__(self, type_gestures, base_dir, n_states = 6, n_samples = 20):
        # Type (unistroke or multistroke)
        self.type_gestures = type_gestures
        # Num states and num samples
        self.n_states = n_states
        self.n_samples = n_samples
        ## Gesture dataset
        # Unistroke
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
                ("arrow", 4 * n_samples), ("caret", 2 * n_samples),
                ("circle", 4*n_samples), ("check", 2*n_samples),
                ("delete_mark", 3*n_samples), ("left_curly_brace", 6*n_samples),
                ("left_sq_bracket", 3*n_samples), ("pigtail", 4*n_samples),
                ("question_mark", 4*n_samples), ("rectangle", 4*n_samples),
                ("right_curly_brace", 6*n_samples), ("right_sq_bracket", 3*n_samples),
                ("star", 5*n_samples), ("triangle", 3*n_samples),
                ("v", 2*n_samples), ("x", 3*n_samples)
            ]
            # baseDir
            self.base_dir = base_dir + '1dollar-dataset/resampled/'
        # TODO add multistroke gestures

        # HMMs
        self.hmms = []
        # Creates models
        parse = Parse(n_states, n_samples)
        for gesture in self.gestures:
            # Gets the hmm
            model = parse.parseExpression(self.type_gestures + "-" + gesture)
            self.hmms.append(model)


    def test(self):
        """
            this method starts test phase
        :return: none
        """
        test_object = test(self.hmms, self.gestures, self.base_dir)
        test_object.all_files()


    # Applies normalize transforms
    def normalizeData(self, sequence):
        # TODO normalize and return the sequence
        return sequences



# Example #
type_gesture = "unistroke"# Tipo di gesture
base_dir = '/home/ale/PycharmProjects/deictic/repository/deictic/'# Path del repository
deictic = Deictic(type_gestures=type_gesture, base_dir=base_dir)# Crea un oggetto di tipo deictic
deictic.test()# Esegui il test (il file viene salvato nel repository con il nome: matrix_confusion_choice.csv
