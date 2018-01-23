from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
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

        print('')
        print(data)
        print('')
        models = json.loads(data)
        print(models['triangle'])
        print('')

        factory = ClassifierFactory(type=TypeRecognizer.offline)
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.states = 16
        factory.spu = 20

        #triangle = factory.createClassifier(triangleExpr);
        #rectangle = factory.createClassifier(rectangleExpr);

        #request.session['models'] = {
        #    'triangle' : 'triangle',
        #    'rectangle': 'rectangle'
        #}

    #sample = request.session['models']['triangle'][0].sample();

    #print(sample)
    #print(numpy.exp(request.session['models']['triangle'][0].log_probability(sample) / len(sample)))
    #print(numpy.exp(request.session['models']['triangle'][0].log_probability(sample) / len(sample)))
    return JsonResponse({'test': 'ok'})
