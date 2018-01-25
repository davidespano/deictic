var scene = new Grid.Scene();
scene.init();
var input = new Utils.StrokeInput(scene.layer);
var feedback = new Grid.Octopocus();
feedback.init(scene.layer);

var deictic = new Utils.Deictic();
deictic.init([
    {name: 'triangle',  model: 'P(0,0) + L(-3,-4) + L(6,0)+ L(-3,4)'},
    {name: 'square', model: 'P(0,0) + L(0,-3) + L(3,0) + L(0, 3) + L(-3,0)'},
    {name: 'delete', model: 'P(0, 0) + L(3, -3) + L(-3, 0) + L(3, 3)'}
]);


input.onStrokeBegin.add(function (event) {
    feedback.start(event);
});

input.onStrokeChange.add(function (event) {

    var descr = [];
     var result = deictic.eval(event.d.buffer);
     for (var i in result){
         var gesture = result[i]
         var max = 0.0;
         var maxIndex = 0;
         for (var j in gesture.parts){
             if (max <= gesture.parts[j].probability){
                 max = gesture.parts[j].probability;
                 maxIndex = j;
             }
         }
         descr.push(
             {name: gesture.name, part: maxIndex, probability: max}
         );
     }

     console.log(descr);
    feedback.update(
        event,
        descr);
});

input.onStrokeEnd.add(function (event) {
    var result = deictic.eval(event.d.buffer);
    var gesture = deictic.recognizedGesture(result, 0.65);
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
    feedback.clear();
});