Recognizing gestures with Deictic
===

Deictic exploits a combination of declarative gesture modelling (i.e. through expression) and Machine Learning for 
supporting a robust but rapidly configurable gesture recognition. The developer models the stroke gestures through
expression composing simple 2D geometric primitives. The underlying recognition engines creates the classifier and 
trains it generating the gesture samples from the those collected for the geometric primitives and shipped with the
library. During the recognition, it provides information on the recognition of the whole gesture and its sub-parts 
as defined by the gesture expression. 

In this test, the class {@link Input.Deictic} object provides a simple API for defining and recognizing gestures with 
the Deictic approach. Please note that the recognition is currently performed at server-side for technical reasons. 

Defining the gesture models
---

The 
 