from __future__ import print_function
# sequence files from deictic
from dataset import Sequence
# tkinter interface
from tkinter import *
# collections
from collections import namedtuple
# config
from config import Config
# matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
# numpy
import numpy as np

class Interface(Frame):

    def __init__(self, title="Window", width=300, height=200):
        # init gui #
        # create main window
        self._master = Tk()
        # set its size
        self._master.geometry(str(width)+"x"+str(height))
        # set its title
        self._master.title(title)
        # init
        Frame.__init__(self, self._master)

# Selected_Point
SelectedPoint = namedtuple('SelectedPoint', 'index point distance_from_click')
# Primitive
Primitive = namedtuple('Primitive', 'num_primitive num_frame')
class Window(Interface):

    # window size
    _width=1920
    _height=1080

    def __init__(self, datasets=None):
        # check
        if not isinstance(datasets, dict):
            raise TypeError
        # init gui
        Interface.__init__(self,title="Deictic", width=self._width, height=self._height)
        # panel image #
        self._left_frame = Frame(self._master, width=self._width-self._width/6, height=self._height, background="bisque")
        self._left_frame.pack(side=LEFT)
        # define a canvas and a fig which will be show in the canvas
        self._fig = Figure(figsize=(16,18))
        self._ax = self._fig.add_subplot(111)
        self._ax.scatter([], [])
        self._ax.set_ylabel("Y", fontsize=14)
        self._ax.set_xlabel("X", fontsize=14)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._left_frame)
        self._canvas.get_tk_widget().pack()
        # panel summary primitives #
        self._right_frame = Frame(self._master, width=self._width/6, height=self._height)
        self._right_frame.pack(side=RIGHT)
        _top_frame = Frame(self._right_frame)
        _bottom_frame = Frame(self._right_frame)

        # labels
        self._filename = Label(_top_frame, text="")
        self._filename.grid(row=0, column=0, padx=(10,60))
        # listbox primitives
        self._list_box=Listbox(_top_frame, width=35, height=40)
        self._list_box.grid(row=1, column=0, pady=(14,14))
        # button next
        self._next_button=Button(_bottom_frame, text="Next Image", fg="black", command=self._next)
        self._next_button.pack(side=LEFT)
        # remove button is used to manage primitives
        self._remove_button = Button(_bottom_frame, text="Remove", fg="red", command=self._remove)
        self._remove_button.pack(side=RIGHT)

        _top_frame.pack(side=TOP,fill=BOTH,expand=1)
        _bottom_frame.pack(side=BOTTOM,fill=BOTH,expand=1)

        #  start application
        self._datasets = datasets
        self._dataset = None
        self._next()
        self.mainloop()

    def _next(self):
        # save list_box contents
        self._save()
        # show new image if available
        if self._dataset != None and len(self._dataset) > 0:
            # clean list box
            self._list_box.delete(0,'end')
            # get sequence
            self._sequence = self._dataset.pop()
            # and plot sequence
            self._plot()
            return
        # change datatet, if available
        if self._datasets:
            next_dataset = self._datasets.popitem() if len(self._datasets)>0 else None
            self._gesture_label = next_dataset[0]
            self._dataset = sum([set.readDataset() for set in next_dataset[1]], []) if next_dataset!=None else None
            return
        # all files have been showed
        self._next_button.config(state="disabled")

    def _remove(self):
        # remove last inserted element
        self._list_box.delete(END)
        # check
        if not self._list_box.size() > 0:
            self._remove_button.config(state="disabled")

    def _save(self):
        """
            add to sequence the references to primitives and save it
        :return:
        """
        if self._list_box.size() > 0:
            # get inserted values
            inserted_values = ([Primitive(int(value[0]),int(value[1])+1) for value
                                        in [(item.split(": ")) for item in self._list_box.get(0,END)]])
            # pull out primitives list from inserted_values
            inserted_values = [inserted_values[0]]+[Primitive(a.num_primitive, a.num_frame-b.num_frame)
                                for a,b in zip(inserted_values[1:],inserted_values)]
            # add new column which describes primitives
            new_column = np.concatenate([np.full(shape=(item.num_frame,1),fill_value=item.num_primitive)
                                            for item in inserted_values])
            self._sequence.points = np.column_stack([self._sequence.getPoints(), new_column])
            # save file
            self._sequence.save(output_dir=Config.baseDir+'deictic/1dollar-dataset/primitives/'
                                           +self._gesture_label+'/')

    def _plot(self):
        # disabled remove button
        self._remove_button.config(state="disabled")
        # get points
        points = self._sequence.getPoints(columns=[0,1])
        # plot and show selected file
        self._fig.tight_layout()
        self._ax.clear()
        self._ax.plot(points[:,0],points[:,1],'o-',picker=self._line_picker)
        for i in range(0, len(points)):
            self._ax.annotate(str(i), (points[i,0], points[i,1]))
        self._filename.config(text=("Gesture: " + self._gesture_label + "\nFilename: " + self._sequence.filename))
        self._canvas.draw()

    def _line_picker(self, line, mouseevent):
        """

        :param line:
        :param mouseevent:
        :return:
        """
        # check
        if mouseevent.xdata is None:
            return False, {}
        # find the closest point to the mouseevent
        b = np.array([mouseevent.xdata, mouseevent.ydata])
        distances = ([SelectedPoint(index, point, np.linalg.norm(point-b))
                      for index,point in enumerate(zip(line.get_xdata(), line.get_ydata()))])
        selected_point = min(distances, key=lambda t: t.distance_from_click)
        self._update(selected_point)
        # throw onclick_event (is necessary)
        return True, {'point':selected_point}
    def _update(self, selected_point):
        # higlight point on figure
        self._ax.plot(selected_point.point[0],selected_point.point[1],'g*')
        # get data and update listbox #
        num_primitive = self._list_box.size()
        num_frame = selected_point.index
        self._list_box.insert(END, str(num_primitive) + ": " + str(num_frame))
        # check
        if self._remove_button['state'] == 'disabled':
            self._remove_button.config(state="normal")