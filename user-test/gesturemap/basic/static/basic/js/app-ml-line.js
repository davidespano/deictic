// init the scene
var scene = new Grid.Scene();
scene.init();

var machineLearning = new Input.MachineLearning();
machineLearning.init()


// init the input
var input = new Input.StrokeInput(scene.layer);

// init the feedback
var lineFeedback = new Grid.LineFeedback();
lineFeedback.init(scene.layer);

input.onStrokeBegin.add(function (event) {



});

input.onStrokeChange.add(function (event) {

 var current = event.d;
    lineFeedback.addPoint(
        event.d.x +
        event.d.bX,
        event.d.y +
        event.d.bY);


});

input.onStrokeEnd.add(function (event) {
    var gesture = machineLearning.recognizedGesture(machineLearning.eval(event.d.buffer), 0.70);

    var x = event.d.x + event.d.bX;
    var y = event.d.y + event.d.bY;

        if(gesture === 'triangle'){scene.setMonster(x, y);}
        if(gesture === 'square'){scene.setTreasure(x, y);}
        if(gesture === 'delete') {scene.clearCell(x,y);}

    lineFeedback.clearLine();



});
