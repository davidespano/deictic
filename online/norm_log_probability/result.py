# Libs
import numpy as np
import sys
import types
# Plot
import matplotlib.pyplot as plt
import matplotlib.pylab as plb

class Result():

    # Methods
    def __init__(self, n_primitives=1):
        self.data_array = []
        # Features
        self.n_primitives = n_primitives

    def get(self):
        """
            converts data_array to string.
        :return:
        """
        data_to_string = []

        for item in self.data_array:
            if isinstance(item, Result):
                data_to_string.append(item.get())
            else:
                data_to_string.append(str(item))

        return data_to_string

    def save(self, path_to_save):
        """
            writes the collected data in the specified file.
        :param path_to_save:
        :return:
        """
        data_to_save = []
        for item in self.data_array:
            if isinstance(item, Result):
                data_to_save.append(item.get())

        np.savetxt(path_to_save+'.txt', data_to_save, fmt='%5s', delimiter=',')

class FrameTestResult(Result):
    """
        FrameTestResult describes the results linked to a single frame. It contains the log probabilites computed for each models in the dataset.
    """

    def __init__(self, test_results, prec_frame = None, n_frame = None, n_primitives=1):
        super(FrameTestResult, self).__init__(n_primitives)
        ### Check validity parameters ###
        if not isinstance(test_results, list):
            raise ValueError('test_results must be an array.')
        if not isinstance(n_frame, int):
            raise ValueError('n_frame must be an integer.')
        ### Features ###
        # Number of frame, if it is known
        self.n_frame = n_frame
        # Reference to the precedent frame
        self.prec_frame = prec_frame
        ## Data structures
        # - log_probabilities contains the log probability value for each model.
        # test_result is an array which its elements have this form: name_model[0] - related log_probability[1]
        self.data_array = test_results
        # - best_log_probability reports which is the model with the highest log probability
        self.best_log_probability = self.__findBestLogProbability()
        # - best_first_derivative reports which is the model with the highest first derivative
        self.best_first_derivative = self.__findBestFirstDerivative()


    def __findBestLogProbability(self):
        """
            find which is the best model in the last result.
        :return: the model with the highest norm log probability value.
        """
        best_model = None
        highest_log_probability = -sys.maxsize# (np.finfo(float).eps)

        # Find the highest model
        for item in self.data_array:
            if item[1] > highest_log_probability:
                best_model = item
                highest_log_probability = item[1]

        return best_model

    def __findBestFirstDerivative(self):
        """

        :return:
        """
        best_model = None
        highest_first_derivative = -sys.maxsize

        # Find the highest model
        if self.prec_frame != None:
            for index in range(len(self.data_array)):
                value_y = 1 # Num_frame
                value_x = self.data_array[index][1] - self.prec_frame.data_array[index][1] # Norm log probability
                der = value_y/value_x # Derivative
                if der > highest_first_derivative:
                    best_model = self.data_array[index]
                    highest_first_derivative = der

        if best_model == None:
            best_model = self.data_array[0]

        return best_model



    # Override
    def get(self):
        """
            converts data_array to string.
        :return:
        """
        # Num frame, norm log probability and model name
        return (str(self.n_frame), str(self.best_log_probability[1]), str(self.best_log_probability[0]))

    def resultsToArray(self):
        """
            converts data_array to string.
        :return:
        """
        data = {}
        for item in self.data_array:
            data[item[0]] = [item[1]]
        return data

class FileTestResult(Result):
    """
        FileTestResult describes the results linked to a single file.
    """

    def __init__(self, file_name, n_primitives = 1):
        super(FileTestResult, self).__init__(n_primitives)
        ### Check validity parameters ###
        if not isinstance(file_name, str):
            raise ValueError('file_name must be a string.')
        ### Features ###
        # The path of file
        self.file_name = file_name
        ## Data Structures
        # dictionary: frame number (key) -> the best primitive (value)
        self.computedSequence = {}
        # This array reports for each models the log probabilitie values during the evolution
        self.testTrend = {}

    def update(self, data_test):
        """
            creates a new FrameTestResult and passes to it the incoming results.
        :param data_test:
        :return:
        """
        # Create a new FrameTestResult and add it into the correct array
        n_frame = len(self.data_array)+1
        prec_frame = None
        if len(self.data_array) > 0:
            prec_frame = self.data_array[-1]
        self.data_array.append(FrameTestResult(data_test, prec_frame=prec_frame, n_frame=n_frame))
        # Update computedSequence array
        #self.computeBestSequence()
        # Update test trend
        self.collectModelsTrend()

    def computeBestSequence(self):
        """
            Analyzes the results of all receveid frames.
            In computedSequence, we reported the evolution of the primitives throughout the file,
            highlighting when and how much frames the hmm changes its status.
        :return: None
        """
        # Get the last best model
        last_best_value_recorded = self.data_array[-1].best_log_probability
        if len(self.computedSequence) > 0:
            last_hmm = self.computedSequence[-1]
            if last_hmm[0] != last_best_value_recorded[0]:
                model_name = last_best_value_recorded[0]
                start_frame = self.data_array[-1].n_frame
                end_frame = start_frame + 1
                during = 1
                self.computedSequence.append([model_name, start_frame, end_frame, during])
            else:
                model_name = last_hmm[0]
                start_frame = last_hmm[1]
                end_frame = self.data_array[-1].n_frame
                during =  end_frame - start_frame
                self.computedSequence[-1]=[model_name, start_frame, end_frame, during]
        else:
            model_name = last_best_value_recorded[0]
            start = 0
            end = 1
            during = 1
            self.computedSequence.append([model_name, start, end, during])


    def collectModelsTrend(self):
        """

        :return:
        """
        for key,value in self.data_array[-1].resultsToArray().items():
            if key not in self.testTrend:
                self.testTrend[key] = []
            self.testTrend[key].append(value)

    def check(self):
        """
            This method analyzes the data include into computedSequence in order to check if primitives have been recognized correctly.
        :return:
            yes (the primitives have been recognized correctly) or false.
        """
        basic_recognized = 1
        # Scan the array, in order to check if the primitives are recognized correctly
        for frame in self.data_array:
            hmm_name = frame.best_log_probability[0]

            if str(basic_recognized) in hmm_name or str(basic_recognized+1) in hmm_name:
                if str(basic_recognized+1) in hmm_name:
                    basic_recognized+=1
            else:
                return False
        # Has been recognized the complete gesture? If yes return true else false
        if basic_recognized == self.n_primitives+1:
            return True


    def plot(self):
        """
            this method collects the log probabilities of each model and creates a plot of these data.
        :return:
        """
        # Get data
        #print(self.file_name)
        fig, ax = plb.subplots(1,1,figsize=(18,20))
        for key,value in self.testTrend.items():
            x = np.arange(len(self.data_array))
            y = np.asarray(value)
            plb.plot(x,y, label=key)
            ax.scatter(x, y)
            for i in range(0, len(value)):
                ax.annotate(str(i), (x[i], y[i]))
        # Title
        plb.title(self.file_name)
        # Legend
        plb.legend(bbox_to_anchor=(.05, 1), loc='best', borderaxespad=0.)
        # x ticks
        plb.xticks(np.arange(min(x), max(x) + 1, 2.0))
        #plb.ylim(-250, 1)
        # Show image
        plb.show()



class DatasetTestResult(Result):
    """
        DatasetTestResult describes the results linked to a whole dataset.
    """

    def __init__(self, n_primitives=1):
        super(DatasetTestResult, self).__init__(n_primitives)
        #### Features ####
        # For each file, test_result reports if the primitives have been recognized correctly or not.
        self.test_result = {}

    # Handlers for managing the test_real_time result related to a speficified file's frame.
    def start(self, file_name):
        """
            handler links to the event "starting to 'fire' a new file".
            :return: None

        """
        # Inizialize dictionary
        self.data_array.append(FileTestResult(file_name))

    def update(self, data_test):
        """
            handler links to the event "firing a new frame".
            :return: None
        """
        self.data_array[-1].update(data_test)

    def stop(self, file_name):
        """
            handler links to the event "file completed"
            :return: None
        """
        self.test_result[file_name] = self.data_array[-1].check()

    # Override
    def save(self, path_to_save):
        """
            saves the data in the specified file.
        :param path_to_save:
        :return:
        """
        for item in self.data_array:
            item.save(path_to_save+item.file_name)
    # Override
    def plot(self, csvDataset = None):
        """
            plots the analized data.
        :return:
        """
        for item in self.data_array:
            item.plot()
            # If csvDataset is not None, plots also the file
            csvDataset.plot(sampleName=item.file_name)

    # --- try ---
    def find(self, file_name):
        """

        :param file_name:
        :return:
        """
        for item in self.data_array:
            if item.file_name == file_name:
                return  item
        return None

    # Shows dataset result
    def showResult(self):
        """

        :return:
        """
        success = 0
        fail = 0
        for value in self.test_result.values():
            if value :
                success+=1
            else:
                fail+=1
        return success, fail
    def findMaxPrimitive(self, file_name, num_primitive, der=False):
        """

        :param file_name:
        :param num_primitive:
        :return:
        """
        file_description = self.find(file_name)
        for frame in file_description.data_array:
            if der:
                if str(num_primitive) in frame.best_first_derivative[0]:
                    return frame.n_frame
            else:
                if str(num_primitive) in frame.best_log_probability[0]:
                    return  frame.n_frame
        return None