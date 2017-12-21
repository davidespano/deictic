
input.onStrokeBegin.add(function (event) {
    lineFeedback.clearLine();
});

input.onStrokeChange.add(function(event){
    lineFeedback.addPoint(
        event.d.x +
        event.d.bX,
        event.d.y +
        event.d.bY);
});

input.onStrokeEnd.add(function(event){

});



