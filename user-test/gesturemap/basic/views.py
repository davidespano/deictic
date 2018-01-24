from django.shortcuts import render

# Create your views here.
from django.http import HttpResponseBadRequest
from django.http import JsonResponse
import numpy as numpy
from django.views.decorators.cache import never_cache
import sys

sys.path.append('/Users/davide/PycharmProjects/deictic')

from gesture import *

import json



def index(request):
    context = {}
    return render(request, 'basic/index.html', context)


@never_cache
def deictic_models(request):
    parser = StringParser()

    if not 'models' in request.session:
        baseDir = '/Users/davide/PycharmProjects/deictic/repository/'
        trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
        arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
        arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'

        data = request.body

        modelDefinitions = json.loads(data)

        factory = ClassifierFactory(type=TypeRecognizer.offline)
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.states = 6
        factory.spu = 20

        models = {}
        for model in modelDefinitions:
            expr = parser.fromString(model['model'])
            if model['name'] == 'rectangle':
                factory.spu = 20
            else:
                factory.spu = 15
            trained = factory.createClassifier(expr)
            models[model['name']] = trained[0]
        request.session['models'] = models

    #sample = request.session['models']['triangle'].sample()

    #print(sample)
    #print(numpy.exp(request.session['models']['triangle'].log_probability(sample) / len(sample)))
    #print(numpy.exp(request.session['models']['rectangle'].log_probability(sample) / len(sample)))
    return JsonResponse({'result': 'ok'})

@never_cache
def deictic_eval(request):

    if 'models' in request.session:
        samples = 40
        models = request.session['models']
        res = []
        data = json.loads(request.body)
        sequence = []
        for point in data:
            sequence.append([
                point['x'],
                - point['y']
            ])

        sequence = numpy.array(sequence).astype(float)
        composite3 = CompositeTransform()
        composite3.addTranform(NormaliseLengthTransform(axisMode=True))
        composite3.addTranform(ScaleDatasetTransform(scale=100))
        composite3.addTranform(CenteringTransform())
        composite3.addTranform(ResampleInSpaceTransform(samples= 3 * samples))


        composite4 = CompositeTransform()
        composite4.addTranform(NormaliseLengthTransform(axisMode=True))
        composite4.addTranform(ScaleDatasetTransform(scale=100))
        composite4.addTranform(CenteringTransform())
        composite4.addTranform(ResampleInSpaceTransform(samples=4 * samples))

        transformed3 = composite3.transform(sequence)
        transformed4 = composite4.transform(sequence)

        res.append({
                'name': 'triangle',
                'prob': models['triangle'].log_probability(transformed3) / len(transformed3)
        })

        res.append({
            'name': 'rectangle',
            'prob': models['rectangle'].log_probability(transformed3) / len(transformed3)
        })
        #for key in models:
        #    prob = models[key].log_probability(transformed) / len(transformed)
        #    el = {}
        #    res.append({
        #        'name': key,
        #        'prob': prob
        #    })

        print(res)
        return JsonResponse({'result': res})

    else:
        return HttpResponseBadRequest


