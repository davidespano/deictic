var scene = new Grid.Scene();
scene.init();
var input = new Input.StrokeInput(scene.layer);
var feedback = new Grid.Octopocus();
feedback.init(scene.layer);

var ml = new Input.MachineLearning();
ml.init();


input.onStrokeBegin.add(function (event) {
    feedback.start(event);
});

input.onStrokeChange.add(function (event) {

    var descr = [
        {name: "triangle", part: 0, probability: 0},
        {name: "square", part: 0, probability: 0},
        {name: "delete", part: 0, probability: 0}
        ];
    var result = ml.eval(event.d.buffer);
    for (var i in result) {
        var gesture = result[i];
        var max = 0.0;
        var maxIndex = 0;
        for (var j in descr) {
            var key = descr[j].name;
            if (gesture.name.includes(key)) {
                if(descr[j].probability <= gesture.probability){
                    descr[j].probability = gesture.probability;
                    if(gesture.name.includes("p1")){
                        descr[j].part = 0;
                    }else if(gesture.name.includes("p2")){
                        descr[j].part = 1;
                    }else if(gesture.name.includes("p3")){
                        descr[j].part = 2;
                    }else if(gesture.name === "square"){
                        descr[j].part = 3
                    }else{
                        descr[j].part = 2;
                    }
                }
            }
        }
    }

    console.log(descr);
    feedback.update(
        event,
        descr);
});

input.onStrokeEnd.add(function (event) {
    var result = ml.eval(event.d.buffer);
    var gesture = ml.recognizedGesture(result, 0.70);
    switch (gesture) {
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