from django.shortcuts import render

# Create your views here.
from django.http import HttpResponseBadRequest
from django.http import JsonResponse
import numpy as numpy
import sys
sys.path.append('/Users/davide/PycharmProjects/deictic')

from gesture import *

import json



def index(request):
    context = {}
    return render(request, 'basic/index.html', context)


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
            trained = factory.createClassifier(expr)
            models[model['name']] = trained[0]
        request.session['models'] = models

    #sample = request.session['models']['triangle'].sample()

    #print(sample)
    #print(numpy.exp(request.session['models']['triangle'].log_probability(sample) / len(sample)))
    #print(numpy.exp(request.session['models']['rectangle'].log_probability(sample) / len(sample)))
    return JsonResponse({'result': 'ok'})


def deictic_eval(request):

    if 'models' in request.session:
        models = request.session['models']
        res = []
        data = json.loads(request.body)
        seq = []
        for point in data:
            seq.append([
                point['x'] * 0.5 - 50,
                point['y'] * 0.5 - 50
            ])

        #seq = models['triangle'].sample()
        print(seq)
        for key in models:
            prob = models[key].log_probability(seq) / len(seq)
            el = {}
            res.append({
                'name': key,
                'prob': prob
            })
        return JsonResponse({'result': res})

    else:
        return HttpResponseBadRequest


