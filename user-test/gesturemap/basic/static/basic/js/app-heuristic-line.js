// init the scene
var scene = new Grid.Scene();
scene.init();

// init the input
var input = new Input.StrokeInput(scene.layer);

// init the feedback
var lineFeedback = new Grid.LineFeedback();

var angleFSMT = new Input.AngleFSM();
var angleFSMS = new Input.AngleFSM();
var angleFSMX = new Input.AngleFSM();

var midRange = 50;
var tolerance = 5;

angleFSMT.init([
   {min: 240 - midRange, max: 240 + midRange},
   {min: 0 -  midRange, max: 0 +  midRange},
   {min: 115 -  midRange, max: 115 +  midRange}
 ], tolerance);

 angleFSMS.init([
    {min: 270 - midRange, max: 270 + midRange},
    {min: 0 -  midRange, max: 0 +  midRange},
    {min: 90 -  midRange, max: 90 +  midRange},
    {min: 180 -  midRange, max: 180 +  midRange}
  ], tolerance);

  angleFSMX.init([
     {min: 315 - midRange, max: 315 + midRange},
     {min: 180 -  midRange, max: 180 +  midRange},
     {min: 45 -  midRange, max: 45 +  midRange}
   ], tolerance);


lineFeedback.init(scene.layer);

input.onStrokeBegin.add(function (event) {
    lineFeedback.clearLine();
     angleFSMT.restart();
     angleFSMS.restart();
     angleFSMX.restart();
});

input.onStrokeChange.add(function (event) {

     var current = event.d;
        lineFeedback.addPoint(
            event.d.x +
            event.d.bX,
            event.d.y +
            event.d.bY);


     var previous = event.d.buffer[event.d.buffer.length - 2];

     // update the FSM
     angleFSMT.push(current, previous);
     angleFSMS.push(current, previous);
     angleFSMX.push(current, previous);
});

input.onStrokeEnd.add(function (event) {
    lineFeedback.clearLine();
    var x = event.d.x + event.d.bX;
    var y = event.d.y + event.d.bY;

    // tests if the FSM is in the error state
    if(angleFSMT.state < 0){
       // error
       console.error("Error");
    }

    // checks if a gesture has been recognized
    if(angleFSMT.state === angleFSMT.states.length - 1){
        console.log("Success");
        scene.setMonster(x, y);

    }
    if(angleFSMS.state === angleFSMS.states.length - 1){
        console.log("Success");
        scene.setTreasure(x, y);

    }
    if(angleFSMX.state === angleFSMX.states.length - 1){
        console.log("Success");
        scene.clearCell(x,y);

    }
});


