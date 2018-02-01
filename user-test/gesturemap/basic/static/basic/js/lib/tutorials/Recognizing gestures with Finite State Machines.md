Recognizing Gestures with Finite State Machines
===

In this user test, the gesture recognition with the Finite State Machines (FSM) is a sample of the heuristic approaches. 
They are quite used in user-interface programming, for instance in 
<a href="https://developer.apple.com/documentation/uikit/touches_presses_and_gestures/implementing_a_custom_gesture_recognizer/about_the_gesture_recognizer_state_machine">
iOS</a>. 

We provide a simplified API for building FSMs able to recognize linear stroke gestures. The heuristic is simply based
on the angle in the goniometric circle defined by the previous and the current stroke point. At each update, the FSM
checks if the angle is contained into the specified range. If so, the recognition continues. Otherwise, the stroke is 
rejected. 

For instance, consider the ideal trajectory of the V gesture in the figure below. During its first part (in red),
the user should move along the 300 degrees direction, visible in the first circle. However, it is impossible for a user
to maintain the direction precisely, therefore we can create a 30-degrees tolerance range around the ideal direction,
which corresponds to the green sector in the first circle (minimum 285, maximum 315). In a similar way, we can build
a tolerance range around the direction for the second part (in blue, minimum 45, maximum 75 degrees). Such ranges can be
found empirically by trial and error. 

<img src="./tutorials/img/5-1-angles.png" alt="A sample grid map" style="width: 800px;"/> 

The recognition state machine for a V gesture is depicted in the following figure. At the beginning it is at the `Start`
state. If the user follows correctly the 300 degrees direction it fires the transition to the `Part 1` state, otherwise
it fires the transition to the `Error` state and the recognition ends.
In the correct case, the FSM remains in the `Part 1` state util either the user misses the direction (`Error` state) or
the stroke starts moving along the 60 degrees direction. In the latter case the FMS fires the transition towards `Part 2`.
Other following movements in the 60 degrees direction maintain the FMS in this state. Again, moving outside the direction
will fire a transition towards the `Error` state. 
If the user ends the stroke while the FSM is in `Part 2` state, the gesture is recognized.

<img src="./tutorials/img/5-2-angles.png" alt="A sample grid map" style="width: 600px;"/> 

The {@link Input.AngleFSM} class simplifies the creation of stroke recognition FSMs. It accepts a configuration object 
including the definition of the angle range for each state and fires the transitions accordingly when the stroke samples
are available. The configuration object consists in an array of range definitions, in which each element defines the
minimum and the maximum angle. The initialization for the V gesture is the following:

````
var angleFSM = Input.AngleFSM();
var midRange = 15;
var tolerance = 5;

angleFSM.init([
   {min: 300 - midRange, max: 300 + midRange},
   {min: 60 -  midRange, max: 60 +  midRange}
 ], tolerance);
 ````


