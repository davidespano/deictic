var scene = new Grid.Scene();
scene.init();
var input = new Utils.StrokeInput(scene.layer);
var lineFeedback = new Grid.LineFeedback();
lineFeedback.init(scene.layer);



var triangleFSM = new Utils.AngleFSM();
triangleFSM.init([
    {min: 210, max: 270},
    {min: -20, max: 20},
    {min: 100, max: 140}
]);

var squareFSM = new Utils.AngleFSM();
squareFSM.init([
    {min: 270 - 20, max: 270 + 20},
    {min: -20, max: 20},
    {min: 90 - 20, max: 90 + 20},
    {min: 180 - 20, max: 180 + 20}
]);


var deleteFSM = new Utils.AngleFSM();
deleteFSM.init([
    {min: 315 -20, max: 315 + 20},
    {min: 180 - 20, max: 180 + 20},
    {min: 45 - 20, max: 45 + 20}
]);

// ------------------ //
input.onStrokeBegin.add(function (event) {
    triangleFSM.restart();
    squareFSM.restart();
    deleteFSM.restart();

});

input.onStrokeChange.add(function (event) {
    var current = event.d;
    var previous = event.d.buffer[event.d.buffer.length - 2];
    triangleFSM.push(current, previous);
    squareFSM.push(current, previous);
    deleteFSM.push(current, previous);
    lineFeedback.addPoint(
        event.d.x +
        event.d.bX,
        event.d.y +
        event.d.bY);
});

input.onStrokeEnd.add(function (event) {
    if (triangleFSM.state == 2) {
        scene.setMonster(
            event.d.bX,
            event.d.bY
        );
        console.log('triangle');
    }

    if(squareFSM.state == 3){
        scene.setTreasure(
            event.d.bX,
            event.d.bY
        );

        console.log('square');
    }

    if (deleteFSM.state == 2) {
        scene.clearCell(
          event.d.bX,
          event.d.bY
        );

        console.log('delete');
    }

    lineFeedback.clearLine();
});

