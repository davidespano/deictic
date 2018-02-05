Registering strokes
===================

In order to ease the gesture tracking and recognition, the class {@link Input.StrokeInput} provides a unified API
for registering both mouse and touch strokes. 

After creating an {@link Input.StrokeInput} instance, it must be linked to the grid map canvas invoking the `init` 
function and passing the `layer` member of the {@link Grid.Scene}.

````
var scene = new Grid.Scene();
scene.init();

// connects the stroke input manager to the grid map
var input = new Input.StrokeInput(scene.layer);
````

Each {@link Input.StrokeInput} instance provides three events (instances of the {@link Input.Event} class) 
for tracking the user's strokes:
* `onStrokeBegin` that notifies when a new stroke begins (e.g., mouse down or touch start);
* `onStrokeChange` that notifies when a stroke has changed (e.g., mouse or touch move);
* `onStrokeEnd` that notifies when a stroke is completed (e.g., mouse up or touch end).

When the user performs a stroke, the {@link Input.StrokeInput} raises first a `onStrokeBegin` event, the a sequence
of `onStrokeChange` events and finally a `onStrokeEnd`. If the user clicks or taps the cell without any movement,
the you will receive only a `onStrokeBegin` and a `onStrokeEnd` event. 

Each event passes to the callback function an `event` object that in the `d` member contains information 
on the stroke evolution:
* `event.d.X` and `event.d.Y` represent the position of the current stroke point (e.g., the current mouse or touch 
position), in the grid cell coordinates (min: 0, max: 200).
* `event.d.bX` and `event.d.bY` represent the coordinates of the top-left corner of the grid cell containing the 
stroke.
* `event.d.buffer` the set of point defining the current stroke. The {@link Input.StrokeInput} object automatically
buffers the points at each update. The last point in the buffer is  (`event.d.X`, `event.d.Y`). The following are a 
set of relevant usage examples.

Reading the current point when the stroke begins
-------------
````
input.onStrokeBegin.add(function(event){
     var currentX = event.d.x;
     var currentY = event.d.y;
});
````

Computing the difference on both axis at each stroke change
--------
````
input.onStrokeChange.add(function(event){
     // we read the previous point using the buffer,
     // the current point is the last one
     var previous = event.d.buffer[event.d.buffer.length - 2];
     var diffX = event.d.x - previous.x;
     var diffY = event.d.y - previous.y;
});
````

Set a monster in the current cell when the stroke ends
---------
````
nput.onStrokeEnd.add(function(event){
    // add the current stroke point coordinates to the top-left cell corner
    var x = event.d.x + event.d.bX;
    var y = event.d.y + event.d.bY;
    
    scene.setMonster(x,y);

});
````


