var scene = new Grid.Scene();
scene.init();
var input = new Utils.StrokeInput(scene.layer);
var lineFeedback = new Grid.LineFeedback();
lineFeedback.init(scene.layer);

//Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)
var deictic = new Utils.Deictic();
deictic.init([
    {name: 'triangle',  model: 'P(0,0) + L(-3,-4) + L(6,0)+ L(-3,4)'},
    {name: 'square', model: 'P(0,0) + L(0,-3) + L(3,0) + L(0, 3) + L(-3,0)'},
    {name: 'delete', model: 'P(0, 0) + L(3, -3) + L(-3, 0) + L(3, 3)'}
]);



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
    var result = deictic.eval(event.d.buffer);
    var gesture = deictic.recognizedGesture(result, 0.70);
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

