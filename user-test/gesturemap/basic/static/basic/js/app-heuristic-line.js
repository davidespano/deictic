// init the scene
var scene = new Grid.Scene();
scene.init();

// init the input
var input = new Input.StrokeInput(scene.layer);

// init the feedback
var lineFeedback = new Grid.LineFeedback();
lineFeedback.init(scene.layer);


input.onStrokeBegin.add(function (event) {

});

input.onStrokeChange.add(function (event) {

});

input.onStrokeEnd.add(function (event) {

});

