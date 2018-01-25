var scene = new Grid.Scene();
scene.init();
var input = new Utils.StrokeInput(scene.layer);
var feedback = new Grid.Octopocus();
feedback.init(scene.layer);

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

input.onStrokeBegin.add(function (event) {
    triangleFSM.restart();
    squareFSM.restart();
    deleteFSM.restart();
    feedback.clear();
    feedback.start(event);
});

input.onStrokeChange.add(function (event) {

    var current = event.d;
    var previous = event.d.buffer[event.d.buffer.length - 2];
    triangleFSM.push(current, previous);
    squareFSM.push(current, previous);
    deleteFSM.push(current, previous);

    var triangleState = triangleFSM.state >= 0 ? triangleFSM.state : 0;
    var triangleProb = (triangleFSM.state + 1) / 3;
    var squareState = squareFSM.state >= 0 ? squareFSM.state : 0;
    var squareProb = (squareFSM.state + 1) / 4;
    var deleteState = deleteFSM.state >= 0 ? deleteFSM.state : 0;
    var deleteProb = (deleteFSM.state + 1) / 3;
    var descr = [
            {name: 'triangle', part: triangleState, probability: Math.max(0, triangleProb)},
            {name: 'square', part: squareState, probability: Math.max(0, squareProb)},
            {name: 'delete', part: deleteState, probability: Math.max(0, deleteProb)}
        ];
    feedback.update(
        event,
        descr);
});

input.onStrokeEnd.add(function (event) {
    if (triangleFSM.state == 2) {
        scene.setMonster(
            event.d.bX,
            event.d.bY
        );
        console.log('triangle');
    }

    if (squareFSM.state == 3) {
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

    feedback.clear();
});