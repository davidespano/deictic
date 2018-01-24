from gesture import *
import numpy as numpy
import matplotlib.pyplot as plt

parser = StringParser()

baseDir = '/Users/davide/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
samples = 60
states = 6

modelDefinitions = [
    {'name': 'triangle', 'model': 'P(0,0) + L(-3,-4) + L(6,0)+ L(-3,4)'},
    {'name': 'rectangle', 'model': 'P(0,0) + L(0,-3) + L(3,0) + L(0, 3) + L(-3,0)'},
    {'name': 'delete', 'model': 'P(0, 0) + L(3, -3) + L(-3, 0) + L(3, 3)'}
]

for definition in modelDefinitions:
    complete = parser.fromString(definition['model'])
    parts = [{'name': str(complete), 'exp': complete, 'hmm': None}]
    current = complete
    while current.is_composite():
        current = current.left
        parts.append(
            {'name': str(current), 'exp': current.clone(), 'hmm': None}
        )
    definition['parts'] = list(reversed(parts))

factory = ClassifierFactory(type=TypeRecognizer.offline)
factory.setLineSamplesPath(trainingDir)
factory.setClockwiseArcSamplesPath(arcClockWiseDir)
factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
factory.states = states

for definition in modelDefinitions:
    for i in range(1, len(definition['parts'])):
        factory.spu = samples / i

        exp = definition['parts'][i]['exp']
        hmm = factory.createClassifier(exp)
        definition['parts'][i]['hmm'] = hmm[0]




#triangle = factory.createClassifier(triangleExpr);
factory.spu = 15
#rectangle = factory.createClassifier(rectangleExpr);


original = [
    [82, -16], [73, -38], [64, -60], [53, -90], [44, -119], [35, -149],
    [44, -154], [89, -152], [108, -151], [129, -150],
    [143, -150], [155, -149], [144, -131], [131, -114],
     [123, -102], [112, -86], [103, -74], [95, -63], [86, -51], [76, -40],
    [76, -40]
]


transform = CompositeTransform()
transform.addTranform(NormaliseLengthTransform(axisMode=True))
transform.addTranform(ScaleDatasetTransform(scale=100))
transform.addTranform(CenteringTransform())
transform.addTranform(ResampleInSpaceTransform(samples=samples))

transformed = transform.transform(numpy.array(original).astype(float))

plotSample = transformed
plt.plot(plotSample[:, 0], plotSample[:, 1])
plt.show()

res = []
for definition in modelDefinitions:
    parts = []
    for i in range(1, len(definition['parts'])):
        prob = definition['parts'][i]['hmm'].log_probability(transformed) / len(transformed)
        parts.append(
            {'name': definition['parts'][i]['name'], 'prob': prob}
        )
        print("{0}: {1}".format(definition['parts'][i]['name'], prob))
    res.append({
        'name': definition['name'], 'parts': parts
    })
print(res)
# print(triangle[0].log_probability(sample) / len(sample))
# print(rectangle[0].log_probability(sample) / len(sample))
