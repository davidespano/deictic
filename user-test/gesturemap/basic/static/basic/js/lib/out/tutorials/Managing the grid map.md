Managing the grid map
====================

The grid map is a simple 3x4 matrix. Each cell may be empty, or it may contain a monster or a treasure. The following
is a sample grid map. 

<img src="./tutorials/img/1-1-sample.png" alt="A sample grid map" style="width: 400px;"/>

In order to create it, it is sufficient to create an instance of the {@link Grid.Scene} class and invoke the `init` 
method. The only requirement is to include a `div id="stage-parent"` and inside it a  `div id="container"` for
supporting the drawing procedure.

````
var scene = new Grid.Scene();
scene.init();
````

You can change the state of a cell in the grid using the `setMonster`, `setTreasure` and `clearCell` methods. 
All of them accept are user-interaction oriented, so they accept as parameter the `X` and `Y` coordinates pointed
by the user in the grid coordinates (starting from its top-left corner). Assuming that the cells are 200x200 px, the
configuration of the sample cell may be obtained with the code that follows. Please note that you can obtain the same 
result with other coordinates of points contained in the same cell.

````
// monster in cell [1, 0], x in [200, 400), y in [0, 100)
scene.setMonster(354, 57);

// treasure in cell [1, 1], x in [200, 400), y in [200, 400)
scene.setTreasure(289, 338);

// monster in cell [1, 2], x in [400, 600), y in [200, 400)
scene.setMonster(357, 230);

// treasure in cell [2, 3], x in [600, 800), y in [400, 600)
scene.setTreasure(713, 551);
````

The events raised by an instance of the {@link Input.StrokeInput} class, which provides a unified API for mouse and 
touch events, provides the stroke updates in cell coordinates, in order to ease its recognition. 
It provides also the coordinates of upper-left corner of the cell with respect to the grid. So, it is sufficient
to sum the corner point to the cell point in order to obtain the coordinates for invoking the grid modifier methods.
The following code snippet allows clearing a cell when the user clicks on it. 

````
// input is an Input.StrokeInput instance
input.onStrokeEnd.add(function(event){
    // add the current stroke point coordinates to the top-left cell corner
    var x = event.d.x + event.d.bX;
    var y = event.d.y + event.d.bY;
    
    scene.clearCell(x,y);

});
````

