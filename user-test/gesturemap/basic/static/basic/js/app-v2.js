var scene = new Grid.Scene();
scene.init();
var input = new Utils.StrokeInput(scene.layer);
var feedback = new Grid.Octopocus();
feedback.init(scene.layer);

input.onStrokeBegin.add(function (event) {
    feedback.clear();
    feedback.start(event);
});

input.onStrokeChange.add(function (event) {
    feedback.update(
        event,
        [
            {name: 'triangle', part: 0, probability: 0.333},
            {name: 'square', part: 0, probability: 0.333},
            {name: 'delete', part: 0, probability: 0.333}
        ]);
});

input.onStrokeEnd.add(function(event){
  feedback.clear();
});