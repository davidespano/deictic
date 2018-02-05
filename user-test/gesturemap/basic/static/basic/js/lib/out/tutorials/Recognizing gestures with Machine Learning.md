Recognizing gestures with Machine Learning
===

In this test, we consider the _1$ gesture recognizer_ [1] as a representative of the Machine Learning techniques for 
gesture recognition. They all require a set of labelled examples for each one of the different gestures 
we need to recognize and they automatically learn to recognize them. 

The class {@link Input.MachineLearning} provides a simplified API for recognizing gestures using a classifier. We 
need first to train it with gesture samples. After that, we can use the dataset for the recognition.

[1] Jacob O. Wobbrock, Andrew D. Wilson, and Yang Li. 2007. Gestures without libraries, toolkits or training: a 
$1 recognizer for user interface prototypes. In Proceedings of UIST 2007, ACM, 159-168.
<a href="https://doi.org/10.1145/1294211.1294238%20">DOI</a>.

Training step
---
The training step consists in providing different executions of all the considered gestures to the classifier. 
In general, the gesture sample should be performed by different people, in order to have a good representation of
their inter-class variability. In this step we will simplify this step, and you will perform 3 samples of each 
considered gesture, using the <a href="http://localhost:8000/basic/training" target="_blank">
dedicated training interface</a>.

<img src="./tutorials/img/6-1-training.png" alt="A sample grid map" style="width: 300px;"/> 

For providing a sample, you need to draw the stroke in the canvas. After that, you need to write the gesture label
in the text field and press the `Add` button. Repeat the operation 3 times for each gesture 
(at least triangle, square, delete).

Once you have provided all the samples, you can collect the training dataset pressing the `Download` button. 
The interface will dump a Javascript array, that you can copy and paste in the `samples` global variable 
(script `samples.js`). **This step is required for compatibility issues with the original 1$ recognition library**.

For initializing the {@link Input.MachineLearning} class, the `samples` variable must be assigned before calling 
the `init` method. 

````
var machineLearning = new Input.MachineLearning();
// after this statement the MachineLearning object is ready to use.
// The called function  reads internally the samples variable.
machineLearning.init()
````

Recognition step
---
Given the stroke point sequence, the classifier associates a probability to each gesture class defined in the training
set. The `eval` method returns such information. The following example writes the probability of the V gesture when
the whole stroke is available. 

````
var scene = new Grid.Scene();
scene.init();
var input = new Input.StrokeInput(scene.layer);

input.onStrokeEnd.add(function (event) {
    var result = machineLearning.eval(event.d.buffer);

    // check the probability of the V gesture
    var prob = 0;
    for(var i in result){
         var gesture = result[i];
         if(gesture.name === 'V'){
             console.log(gesture.probability);
         }
    }
});
````

In many cases, we consider as the recognized gesture the one having the higher probability. In addition, we threshold
its probability value in order to avoid low values. For instance, if the maximum probability is 0.25 we consider the
not recognized. The {@link Input.MachineLearning} object provides the `recognizedGesture` utility method for searching 
the maximum probability and threshold it. The method takes as input the object returned by `eval`.

````
// get the most likely gesture at the end of the stroke performance.
input.onStrokeEnd.add(function (event) {
    var gesture = machineLearning.recognisedGesture(machineLearning.eval(event.d.buffer), 0.70);
});
````

