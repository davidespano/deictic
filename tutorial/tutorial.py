from gesture import modellingExpression, datasetExample
from test import Test

'''
    shows how to describe a gesture with Deictic and create its model.
'''
# describe gesture swipe_right through deictic
swipe_right = {
        'swipe_right':
            [
                Point(0,0)+Line(3,0)
            ]
        }
# create deictic model for swipe_right
hmm_swipe_right = modellingExpression.ModelFactory(swipe_right)

'''
    
'''
gesture_hmms = datasetExample.datasetExpressions.returnExpressions(datasetExample.datasetExpressions.TypeDataset.dollar1_unistroke)
results = Test.getInstance().offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=none)
results.plot()

'''
    
'''
# describe gesture swipe_right through deictic
gesture_expressions = {
    'swipe_right':
        [
            Point(0,0)+Line(3,0)
        ],
    'swipe_left':
        [
            Point(3,0)+Line(-3,0)
        ]
}
# create deictic model for swipe_right
results = Test.getInstance().offlineTestExpression(gesture_expressions=gesture_expressions, gesture_datasets=none)
