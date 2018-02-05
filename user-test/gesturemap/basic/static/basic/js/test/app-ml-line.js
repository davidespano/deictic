var scene = new Grid.Scene();
scene.init();
var input = new Input.StrokeInput(scene.layer);
var lineFeedback = new Grid.LineFeedback();
lineFeedback.init(scene.layer);
var ml = new Input.MachineLearning();
ml.init();

// ------------------ //
input.onStrokeBegin.add(function (event) {


});

input.onStrokeChange.add(function (event) {

    lineFeedback.addPoint(
        event.d.x +
        event.d.bX,
        event.d.y +
        event.d.bY);
});

input.onStrokeEnd.add(function (event) {
    var result = ml.eval(event.d.buffer);
    var gesture = ml.recognizedGesture(result, 0.70);
    switch(gesture){
        case "triangle":
            scene.setMonster(
            event.d.bX,
            event.d.bY
        );
            break;

        case "square":
            scene.setTreasure(
            event.d.bX,
            event.d.bY
        );
            break;

        case "delete":
            scene.clearCell(
            event.d.bX,
            event.d.bY
        );
            break;
    }
    lineFeedback.clearLine();
});

