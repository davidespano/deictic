from __future__ import print_function
# sequence files from deictic
from dataset import Sequence
# tkinter interface
from tkinter import *
from tkinter.font import Font, nametofont
# recordtype
from recordclass import recordclass
# itemgetter
from operator import itemgetter
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
# copy
from copy import deepcopy
# numpy
import numpy as np
# file
import os

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
SelectedPoint = recordclass('SelectedPoint', 'index point distance_from_click')
# Primitive
Primitive = recordclass('Primitive', 'num_primitive num_frame')
class Window(Interface):

    # window size
    _width=1920
    _width_button = 8
    _height=1080
    _height_button = 2

    def __init__(self, datasets=None):
        # check
        if not isinstance(datasets, dict):
            raise TypeError
        # init gui
        Interface.__init__(self,title="Deictic", width=self._width, height=self._height)
        # fonts
        self._medium_font = Font(root=self, family='Helvetica', size=10, weight='bold')
        self._large_font = Font(root=self, family='Helvetica', size=14, weight='bold')

        # panel image #
        self._left_frame = Frame(self._master, width=self._width-self._width/6, height=self._height, background="bisque")
        self._left_frame.pack(side=LEFT)
        # define a canvas and a fig which will be show in the canvas
        self._fig = Figure(figsize=(16,18))
        self._ax = self._fig.add_subplot(111)
        self._ax.scatter([], [])
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._left_frame)
        self._fig.canvas.mpl_connect('key_press_event', self._keyPress)
        self._canvas.get_tk_widget().pack()

        # panel summary primitives #
        self._right_frame = Frame(self._master, width=self._width/6, height=self._height)
        self._right_frame.pack(side=RIGHT)
        _top_frame_right = Frame(self._right_frame)
        _top_frame_right.pack(side=TOP,fill=BOTH,expand=1)
        _bottom_frame_right = Frame(self._right_frame)
        _bottom_frame_right.pack(side=BOTTOM,fill=BOTH,expand=1)

        # labels
        self._feedback = Label(_top_frame_right, text="")
        self._feedback.grid(row=0, column=0, padx=(10,10), pady=(25,25))
        self._info = Label(_top_frame_right, text="")
        self._info.grid(row=1, column=0, padx=(10,10), pady=(25,25))
        self._filename = Label(_top_frame_right, text="")
        self._filename.grid(row=2, column=0, padx=(10,10), pady=(25,25))
        # listbox primitives
        self._list_box=Listbox(_top_frame_right, width=35, height=15, font=self._medium_font, selectmode="multiple")
        self._list_box.grid(row=3, column=0, padx=(30,30))
        # input
        frame_input = Frame(_top_frame_right)
        frame_input.grid(row=4, column=0, padx=(30,30))
        self._input = Entry(frame_input)
        self._input.grid(row=0, column=0)
        self._input_button = Button(frame_input, text="Insert", fg="black",
                              command=self._newItem, height=self._height_button, width=self._width_button)
        self._input_button.grid(row=0, column=1)

        # button next
        self._next_button=Button(_bottom_frame_right, text="Next Image", fg="black",
                                 command=self._next, height=self._height_button, width=self._width_button)
        self._next_button.grid(row=0,column=0, padx=(20,0), pady=(50,50))
        # button save
        self._save_button = Button(_bottom_frame_right, text="Save", fg="green", state="disabled",
                                     command=self._save, height=self._height_button, width=self._width_button)
        self._save_button.grid(row=0,column=1, pady=(50,50))
        # remove button is used to manage primitives
        self._remove_button = Button(_bottom_frame_right, text="Remove", fg="red", state="disabled",
                                     command=self._remove, height=self._height_button, width=self._width_button)
        self._remove_button.grid(row=0,column=2, padx=(0,20),pady=(50,50))
        #  start application
        self._datasets = datasets
        self._dataset = None
        self._next()
        self.mainloop()

    def _next(self):
        # save
        self._save()
        # delete old listbox
        for i in range(self._list_box.size()):
            self._delete(index=0)
        # plot file, if it is avaiable
        if self._findNextFile():
            self._plot()
            return
        # otherwise change datatet, if available
        if self._datasets:
            next_dataset = self._datasets.popitem() if len(self._datasets)>0 else None
            self._gesture_label = next_dataset[0]
            self._dataset = sum([set.readDataset() for set in next_dataset[1][1]], []) if next_dataset!=None else None
            self._num_primtivies = (next_dataset[1][0] if next_dataset!=None else None)
            return
        # all files have been showed
        self._next_button.config(state="disabled")
    def _findNextFile(self):
        # find the next image
        if self._dataset != None and len(self._dataset) > 0:
            self._sequence = self._dataset.pop(0)
            # get sequence
            return True if not os.path.isfile(Config.baseDir+'deictic/1dollar-dataset/primitives/'+
                                          self._gesture_label+"/"+self._sequence.filename) \
                else self._findNextFile()
        return False

    def _newItem(self):
        """
            this handler manages the input recieved from entry widget. On one hand, it provides to split and verify input,
            and in the other hand to update the list of primitives.
        :return:
        """
        # check
        if len(self._input.get()) > 0:
            try:
                new_primitive = Primitive(num_primitive=self._list_box.size(), num_frame=int(self._input.get()))
                #num_frame = int(self._input.get()) #[int(item) for item in self._input.get().split(" ")]
                #new_primitive = Primitive(values[0],values[1])
            except:
                print("Error! Insert two integers in the entry widget!")
                return
            # clear input
            self._input.delete(0,'end')
            # update list and figure
            self._insert(new_primitive)
            self._change_point(index=new_primitive.num_frame, style_to_plot="ro")
    def _remove(self):
        # remove the selected elements then update plot and lists #
        # get selected elements and create a new Primitive array
        elements = reversed([Primitive(int(item[0]),int(item[1])) for item in
                    [element.split(": ") for element in
                    [self._list_box.get(index) for index in self._list_box.curselection()]]])

        for element in elements:
            # delete selected element
            self._delete(element.num_primitive)
            # delete highlighted point
            self._change_point(element.num_frame, "co")
            # and update list
            for index in range(element.num_primitive,self._list_box.size()):
                temp_item = self._list_box.get(index).split(': ')
                update_item = Primitive(int(temp_item[0])-1,int(temp_item[1]))
                self._delete(index)
                self._insert(item=update_item)

        if not self._list_box.size() > 0:
            self._config_button(buttons=[self._remove_button,self._save_button],states=["disabled","disabled"])
    def _delete(self, index):
        """
            delete an item, with the specify index, from the listbox widget.
        :param index:
        :return:
        """
        # check
        if isinstance(index,str):
            index = int(index)
        try:
            self._list_box.delete(index)
        except:
            print("Index out of boundary")
    def _insert(self, item):
        """
            provide to insert a new item in the listbox widget
        :param item:
        :return:
        """
        # check input
        if not isinstance(item, Primitive):
            raise TypeError("Item must be a Primitive object")
        # check whether the new item has been just inserted #
        # get latest item
        if self._list_box.size() > 0:
            old_item = Primitive(num_primitive=-1, num_frame=int(self._list_box.get(END).split(" ")[-1]))
            if item.num_frame <= old_item.num_frame:
                return
        # insert new item
        self._list_box.insert(item.num_primitive, str(item.num_primitive) + ": " + str(item.num_frame))
        # check buttons state
        if self._remove_button['state'] == 'disabled':
            self._config_button(buttons=[self._remove_button,self._save_button],states=["normal","normal"])

    def _save(self):
        """
            get from the widget the list of primitives and add to the file the references about primitives then save it.
        :return:
        """
        if self._list_box.size() > 0:
            if self._list_box.size()+1 == self._num_primtivies:
                self._insert(item=Primitive(self._list_box.size(),len(self._sequence.getPoints())-1))
            if self._list_box.size() < self._num_primtivies:
                raise Exception("You must select "+str(self._num_primtivies)+" primitives.")

            try:
                # get inserted values
                inserted_values = [Primitive(int(item[0]),int(item[1])+1) for item in
                                   [element.split(": ") for element in self._list_box.get(0,END)]]
                # determine the lenght of each primitive (t.num_frame - t-1.num_frame)
                temp = deepcopy(inserted_values)
                for a,b in zip(inserted_values[1:],temp):
                    a.num_frame = a.num_frame-b.num_frame
                # add new column which describes primitives
                new_column = np.concatenate([np.full(shape=(item.num_frame,1),fill_value=item.num_primitive)
                                                for item in inserted_values])
                self._sequence.points = np.column_stack([self._sequence.getPoints(), new_column])
                # save file
                self._sequence.save(output_dir=Config.baseDir+'deictic/1dollar-dataset/primitives/'
                                               +self._gesture_label+'/')
                message = " has been saved"
                color = "green"
            except ValueError as e:
                message = " has not been saved"
                color = "red"
                print(e)

            # notify to user save result
            self._feedback.config(text=(self._sequence.filename+message),
                              font=self._medium_font, fg=color)

    def _plot(self):
        print(self._sequence.filename+" is showed - "+str(len(self._sequence.getPoints())))
        # get points
        points = self._sequence.getPoints(columns=[0,1])
        # plot and show selected file
        self._fig.tight_layout()
        self._ax.clear()
        self._ax.plot(points[:,0],points[:,1],'o-',picker=self._line_picker)
        for i in range(0, len(points)):
            self._ax.annotate(str(i), (points[i,0], points[i,1]))
        # show name of the plotted file
        self._filename.config(text=("File: " + self._sequence.filename),
                              font=self._large_font)
        # show how many files require user intervation
        self._info.config(text=("Label: "+self._gesture_label + " - size: " + str(len(self._dataset))+
                                "\nLenght: "+str(len(points)-1)),
                          font=self._medium_font)
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
        self._change_point(selected_point.index, "ro")
        # insert the selected point in the listbox #
        self._insert(item=Primitive(self._list_box.size(), selected_point.index))
        # re-instate remove button
        if self._remove_button['state'] == 'disabled':
            self._config_button(buttons=[self._remove_button,self._save_button],states=["normal","normal"])
    def _keyPress(self, event):
        if event.key == " ":
            self._next()
    def _change_point(self, index, style_to_plot):
        point = self._sequence.getPoints(columns=[0,1])[index]
        self._ax.plot(point[0],point[1],style_to_plot)
        self._canvas.draw()

    @staticmethod
    def _config_button(buttons=[], states=[]):
        # check
        if not isinstance(buttons,list) and all([isinstance(button,Button) for button in buttons]):
            raise TypeError
        if not isinstance(states,list) and all([state in ["normal","disabled"] for state in states]):
            raise TypeError
        if not len(buttons)==len(states):
            raise TypeError
        # set new states
        for button,state in zip(buttons,states):
            button.config(state=state)