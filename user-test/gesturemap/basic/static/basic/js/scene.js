var width = 1000;
var height = 800;

var stage = new Konva.Stage({
    container: 'container',
    width: width,
    height: height
});

var layer = new Konva.Layer();
stage.add(layer);

var BOX_SIZE = 200;
var box;
// generate boxes
for (var ix = 0; ix < width / BOX_SIZE; ix++) {
    for (var iy = 0; iy < height / BOX_SIZE; iy++) {
        box = new Konva.Rect({
            x: ix * BOX_SIZE,
            y: iy * BOX_SIZE,
            width: BOX_SIZE - 1,
            height: BOX_SIZE - 1,
            fill: 'white',
            stroke: 'lightgray'
        });
        layer.add(box);
    }
}


 var redLine = new Konva.Line({
      points: [5, 70, 140, 23, 250, 60, 300, 20],
      stroke: 'red',
      strokeWidth: 15,
      lineCap: 'round',
      lineJoin: 'round',

    });

layer.add(redLine);

layer.draw();


layer.on('mousedown', function(event){
    if(event.evt.button === 0){
        redLine.points().splice(0, redLine.points().length);
        redLine.points().push(event.evt.x, event.evt.y);
        layer.draw();
    }
});

layer.on('mousemove', function(event){
    if(event.evt.buttons === 1 &&
        event.evt.button === 0 &&
        (event.evt.movementX > 0 && event.evt.movementY > 0)) {
        //console.log(redLine.points())
        redLine.points().push(event.evt.x, event.evt.y);
        redLine.draw();
    }
});

function fitStageIntoParentContainer() {
    var container = document.querySelector('#stage-parent');

    // now we need to fit stage into parent
    var containerWidth = container.offsetWidth;
    // to do this we need to scale the stage
    var scale = containerWidth / width;


    stage.width(width * scale);
    stage.height(height * scale);
    stage.scale({x: scale, y: scale});
    stage.draw();
}


fitStageIntoParentContainer();
// adapt the stage on any window resize
window.addEventListener('resize', fitStageIntoParentContainer);