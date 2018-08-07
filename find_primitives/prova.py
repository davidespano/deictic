from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
import tkinter as tk
from numpy.random import rand


ex = 6

if ex == 0:
    mode = 2

    if mode == 0:  # simple picking, lines, rectangles and text
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title('click on points, rectangles or text', picker=True)
        ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
        line, = ax1.plot(rand(100), 'o', picker=5)  # 5 points tolerance

        # pick the rectangle
        bars = ax2.bar(range(10), rand(10), picker=True)
        for label in ax2.get_xticklabels():  # make the xtick labels pickable
            label.set_picker(True)

        def onpick1(event):
            if isinstance(event.artist, Line2D):
                thisline = event.artist
                xdata = thisline.get_xdata()
                ydata = thisline.get_ydata()
                ind = event.ind
                print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
            elif isinstance(event.artist, Rectangle):
                patch = event.artist
                print('onpick1 patch:', patch.get_path())
            elif isinstance(event.artist, Text):
                text = event.artist
                print('onpick1 text:', text.get_text())

        fig.canvas.mpl_connect('pick_event', onpick1)

    if mode == 1:  # picking with a custom hit test function
        # you can define custom pickers by setting picker to a callable
        # function.  The function has the signature
        #
        #  hit, props = func(artist, mouseevent)
        #
        # to determine the hit test.  if the mouse event is over the artist,
        # return hit=True and props is a dictionary of
        # properties you want added to the PickEvent attributes

        def line_picker(line, mouseevent):
            """
            find the points within a certain distance from the mouseclick in
            data coords and attach some extra attributes, pickx and picky
            which are the data points that were picked
            """
            if mouseevent.xdata is None:
                return False, dict()
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            maxd = 0.05
            d = np.sqrt((xdata - mouseevent.xdata)**2. + (ydata - mouseevent.ydata)**2.)

            ind = np.nonzero(np.less_equal(d, maxd))
            if len(ind):
                pickx = np.take(xdata, ind)
                picky = np.take(ydata, ind)
                props = dict(ind=ind, pickx=pickx, picky=picky)
                return True, props
            else:
                return False, dict()

        def onpick2(event):
            print('onpick2 line:', event.pickx, event.picky)

        fig, ax = plt.subplots()
        ax.set_title('custom picker for line data')
        points_x = [x*1 for x in rand(100)]
        points_y = [x*1 for x in rand(100)]
        line, = ax.plot(points_x, points_y, 'o', picker=line_picker)
        fig.canvas.mpl_connect('pick_event', onpick2)

    if mode==2:
        def onlick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))

            b = np.array([event.xdata, event.ydata])
            distances = ([(point, np.linalg.norm(point-b)) for point in points])
            selected_point = min(distances, key=lambda t: t[1])
            print(selected_point)

        fig,ax = plt.subplots()
        points = []
        x = [x*100 for x in np.random.rand(2)]
        y = [y*100 for y in np.random.rand(2)]
        points = zip(x,y)
        ax.plot(x,y,'o')
        cid = fig.canvas.mpl_connect('button_press_event', onlick)


    plt.show()

if ex == 1:
    import matplotlib
    matplotlib.use('TkAgg')
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from tkinter import *

    class mclass:
        def __init__(self,  window):
            self.window = window
            self.box = Entry(window)
            self.button = Button (window, text="check", command=self.plot)
            self.box.pack ()
            self.button.pack()
        def plot (self):
            x=np.array ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            v= np.array ([16,16.31925,17.6394,16.003,17.2861,17.3131,19.1259,18.9694,22.0003,22.81226])
            p= np.array ([16.23697,     17.31653,     17.22094,     17.68631,     17.73641 ,    18.6368,
                          19.32125,     19.31756 ,    21.20247  ,   22.41444   ,  22.11718  ,   22.12453])
            fig = Figure(figsize=(6,6))
            a = fig.add_subplot(111)
            a.scatter(v,x,color='red')
            a.plot(p, range(2 +max(x)),color='blue')
            a.invert_yaxis()
            a.set_title ("Estimation Grid", fontsize=16)
            a.set_ylabel("Y", fontsize=14)
            a.set_xlabel("X", fontsize=14)
            canvas = FigureCanvasTkAgg(fig, master=self.window)
            canvas.get_tk_widget().pack()
            canvas.draw()
    window= Tk()
    start= mclass (window)
    window.mainloop()

if ex == 2:
    def startgame():

        pass

    mw = tk.Tk()

    #If you have a large number of widgets, like it looks like you will for your
    #game you can specify the attributes for all widgets simply like this.
    mw.option_add("*Button.Background", "black")
    mw.option_add("*Button.Foreground", "red")

    mw.title('The game')
    #You can set the geometry attribute to change the root windows size
    mw.geometry("500x500") #You want the size of the app to be 500x500
    mw.resizable(0, 0) #Don't allow resizing in the x or y direction

    back = tk.Frame(master=mw,bg='black')
    back.pack_propagate(0) #Don't allow the widgets inside to determine the frame's width / height
    back.pack(fill=tk.BOTH, expand=1) #Expand the frame to fill the root window

    #Changed variables so you don't have these set to None from .pack()
    go = tk.Button(master=back, text='Start Game', command=startgame)
    go.pack()
    close = tk.Button(master=back, text='Quit', command=mw.destroy)
    close.pack()
    info = tk.Label(master=back, text='Made by me!', bg='red', fg='black')
    info.pack()

    mw.mainloop()

if ex == 3:
    root = tk.Tk()
    frame1 = tk.Frame(root, width=100, height=100, background="bisque")
    frame2 = tk.Frame(root, width=50, height = 50, background="#b22222")

    frame1.pack(fill=None, expand=False)
    frame2.place(relx=.5, rely=.5, anchor="c")

    root.mainloop()

if ex == 4:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np


    class Disk:


        def __init__(self, center, radius, myid = None, figure=None, axes_object=None):
            """
            @ARGS
            CENTER : Tuple of floats
            RADIUS : Float
            """
            self.center = center
            self.radius = radius
            self.fig    = figure
            self.ax     = axes_object
            self.myid   = myid
            self.mypatch = None


        def onpick(self,event):
            if event.artist == self.mypatch:
                print ("You picked the disk ", self.myid, "  with Center: ", self.center, " and Radius:", self.radius)


        def mpl_patch(self, diskcolor= 'orange' ):
            """ Return a Matplotlib patch of the object
            """
            self.mypatch = mpl.patches.Circle( self.center, self.radius, facecolor = diskcolor, picker=1 )

            #if self.fig != None:
            #self.fig.canvas.mpl_connect('pick_event', self.onpick) # Activate the object's method

            return self.mypatch

    def on_pick(disks, patches):
        def pick_event(event):
            for i, artist in enumerate(patches):
                if event.artist == artist:
                    disk = disks[i]
                    print ("You picked the disk ", disk.myid, "  with Center: ", disk.center, " and Radius:", disk.radius)
        return pick_event

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click on disks to print out a message')

    disk_list = []
    patches = []

    disk_list.append( Disk( (0,0), 1.0, 1, fig, ax   )   )
    patches.append(disk_list[-1].mpl_patch())
    ax.add_patch(patches[-1])

    disk_list.append( Disk( (3,3), 0.5, 2, fig, ax   )   )
    patches.append(disk_list[-1].mpl_patch())
    ax.add_patch(patches[-1])

    disk_list.append( Disk( (4,9), 2.5, 3, fig, ax   )   )
    patches.append(disk_list[-1].mpl_patch())
    ax.add_patch(patches[-1])

    pick_handler = on_pick(disk_list, patches)

    fig.canvas.mpl_connect('pick_event', pick_handler) # Activate the object's method

    ax.set_ylim(-2, 10);
    ax.set_xlim(-2, 10);

    plt.show()

if ex == 5:
    class MyPlot(object):
        def __init__(self, parent=None):
            super(self.__class__, self).__init__()

        def makePlot(self):
            self.fig = plt.figure('Test', figsize=(10, 8))
            ax = plt.subplot(111)
            x = range(0, 100, 10)
            y = (5,)*10
            ax.plot(x, y, '-', color='red')
            ax.plot(x, y, 'o', color='blue', picker=5)
            self.highlight, = ax.plot([], [], 'o', color='yellow')
            self.cid = plt.connect('pick_event', self.onPick)
            plt.show()

        def onPick(self, event=None):
            this_point = event.artist
            x_value = this_point.get_xdata()
            y_value = this_point.get_ydata()
            ind = event.ind
            self.highlight.set_data(x_value[ind][0],y_value[ind][0])
            self.fig.canvas.draw_idle()

    app = MyPlot()
    app.makePlot()

if ex == 6:
    from tkinter import *

    def key(event):
        print("pressed", repr(event.char))

    def callback(event):
        frame.focus_set()
        print("clicked at", event.x, event.y)

    root = Tk()
    frame = Frame(root, width=100, height=100)
    frame.bind("<Key>", key)
    frame.bind("<Button-1>", callback)
    frame.pack()
    root.mainloop()