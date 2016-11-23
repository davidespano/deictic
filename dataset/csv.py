import os


class CsvDataset:


    def __init__(self, dir):
        self.filenames = os.listdir(dir)
        self.dirname = dir
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.i < len(self.filenames):
            if self.filenames[self.i].endswith('.csv'):
                self.i += 1
                return self.filenames[self.i - 1]
            else:
                self.i+= 1
        raise StopIteration