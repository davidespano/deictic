var scene = new Grid.Scene();
scene.init();
var input = new Utils.StrokeInput(scene.layer);
var lineFeedback = new Grid.LineFeedback();
lineFeedback.init(scene.layer);

//Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)
var deictic = new Utils.Deictic();
deictic.init([
    {name: 'triangle',  model: 'P(0,0) + L(-3,-4) + L(6,0)+ L(-3,4)'},
    {name: 'rectangle', model: 'P(0,0) + L(0,-3) + L(4,0) + L(0, 3) + L(-4,0)'}
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
    console.log(result);
    lineFeedback.clearLine();
});

