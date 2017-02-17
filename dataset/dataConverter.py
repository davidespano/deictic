import os
import csv
from lxml import etree
from .csvDataset import DatasetIterator

class UnicaConverter:

    def create_deictic_dataset(self, inputBase, outputBase):
        sub = ['arc3Dleft', 'pigtail', 'spiral',
               'arc3Dright', 'poly3Dxyz', 'square-braket-left',
                'caret', 'poly3Dxzy', 'square-braket-right',
               'check', 'poly3Dyxz', 'star',
               'circle', 'poly3Dyzx', 'triangle',
               'curly-braket-left', 'poly3Dzxy', 'v',
               'curly-braket-right', 'poly3Dzyx', 'x',
               'delete', 'rectangle', 'zig-zag',
               'left', 'right'
               ]
        for name in sub:
            if not os.path.exists(outputBase + '/' + name):
                os.makedirs(outputBase + '/' + name)
            self.replace_csv(inputBase + '/' + name, outputBase + '/' + name)

    # Replace CSV
    # Is used to change the format csv files. It is necessary if files don't have commas or spaces.
    def replace_csv(self, inputDir, outputDir):
        # For each files in the directory
        for file in os.listdir(inputDir):
            # Open and write file
            with open(inputDir + '/' + file) as fin, open(outputDir + '/' + file, 'w') as fout:
                o = csv.writer(fout)
                for line in fin:
                    o.writerow(line.split())


class Dollar1Converter:
    # Xml to CSV
    # Converts input gesture xml files to csv files
    def xml_to_csv(self, inputDir, outputDir, xsltPath):
        iterator = DatasetIterator(inputDir, '.xml')
        for file in iterator:
            data = open(xsltPath)
            xslt_content = data.read()
            xslt_root = etree.XML(xslt_content)
            dom = etree.parse(inputDir + '/'+ file)
            transform = etree.XSLT(xslt_root)
            result = transform(dom)
            f = open(outputDir + '/' + file[:-4] + '.csv', 'w')
            f.write(str(result))
            f.close()
        return

    def create_deictic_dataset(self, inputBase, outputBase):
        sub = ['arrow', 'caret', 'check', 'circle', 'delete', 'left_curly_brace', 'left_sq_bracket',
               'pigtail', 'question_mark', 'rectangle', 'right_curly_brace', 'right_sq_bracket', 'star',
               'triangle', 'v', 'x']
        xsltPath = inputBase + '/' + 'conversion.xslt'
        for name in sub:
            if not os.path.exists(outputBase + '/' + name):
                os.makedirs(outputBase + '/' + name)
            self.xml_to_csv(inputBase + '/' + name, outputBase + '/' + name, xsltPath)
