var width = 800;
var height = 600;

var stage = new Konva.Stage({
    container: 'container',
    width: width,
    height: height
});

var redLine = new Konva.Line({
    points: [],
    stroke: 'red',
    strokeWidth: 10,
    lineCap: 'round',
    lineJoin: 'round',

});

var layer = new Konva.Layer();
var imgLayer = new Konva.Layer();
stage.add(layer);
stage.add(imgLayer);



var imageObj = new Image();
imageObj.onload = function () {

    var yoda = new Konva.Image({
        x: 50,
        y: 50,
        image: imageObj,
        width: 106,
        height: 118
    });

    // add the shape to the layer
    layer.add(yoda);
    layer.draw();
};
imageObj.src = '/static/basic/img/monster.png';


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


layer.add(redLine);

layer.draw();

var input = new Deictic.StrokeInput(layer);
var lineFeedback = new Deictic.LineFeedback(layer, redLine);


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

function debounce(func, wait, immediate) {
    var timeout;
    return function () {
        var context = this, args = arguments;
        var later = function () {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        var callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
};


//fitStageIntoParentContainer();
// adapt the stage on any window resize
//window.addEventListener('resize', fitStageIntoParentContainer);