from django.shortcuts import render

# Create your views here.
from django.http import HttpResponseBadRequest
from django.http import JsonResponse
from django.http import HttpResponseNotFound
import numpy as numpy
from django.views.decorators.cache import never_cache
import sys

sys.path.append('/Users/davide/PycharmProjects/deictic')

from gesture import *

import json


def index(request, condition, feedback):
    script = None
    if (condition == 'heuristic' or condition == 'ml' or condition == 'deictic') and\
            (feedback == 'line' or feedback == 'octo'):
        script = "app-{0}-{1}.js".format(condition, feedback)
        context = {'condition': condition, 'feedback': feedback}
        return render(request, 'basic/index.html', context)
    else:
        return HttpResponseNotFound('<h1> File not found </h1>')



def training(request):
    context = {}
    return render(request, 'basic/training.html', context)


@never_cache
def deictic_models(request):
    parser = StringParser()
    samples = 20
    states = 6

    if not 'models' in request.session:
        baseDir = '/Users/davide/PycharmProjects/deictic/repository/'
        trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
        arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
        arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'

        data = request.body

        modelDefinitions = json.loads(data)

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
                factory.spu = samples

                exp = definition['parts'][i]['exp']
                hmm = factory.createClassifier(exp)
                definition['parts'][i]['hmm'] = hmm[0]

        request.session['models'] = modelDefinitions

    return JsonResponse({'result': 'ok'})

@never_cache
def deictic_eval(request):

    if 'models' in request.session:
        samples = 60
        modelDefinitions = request.session['models']
        res = []
        data = json.loads(request.body)
        sequence = []
        for point in data:
            sequence.append([
                point['x'],
                - point['y']
            ])

        sequence = numpy.array(sequence).astype(float)
        transform = CompositeTransform()
        transform.addTranform(NormaliseLengthTransform(axisMode=False))
        transform.addTranform(ScaleDatasetTransform(scale=100))
        transform.addTranform(CenteringTransform())
        transform.addTranform(ResampleInSpaceTransform(samples=samples))

        transformed = transform.transform(numpy.array(sequence).astype(float))

        res = []
        for definition in modelDefinitions:
            #print(definition['name'])
            parts = []
            for i in range(1, len(definition['parts'])):
                prob = definition['parts'][i]['hmm'].log_probability(transformed) / len(transformed)
                pretty = 0.9468093 + 0.01373144 * prob + 0.00005137985*prob* prob;
                if prob < -130:
                    pretty = 0;
                if prob > 7:
                    prob = 1.0;
                parts.append(
                    {'name': definition['parts'][i]['name'], 'prob': round(pretty, 4), 'logprob': round(prob, 2)}
                )
                #print("{0}: {1} {2} {3}".format(
                #    definition['parts'][i]['name'],
                #    numpy.exp(prob),
                #    prob,
                #    pretty)
                #)
            res.append({
                'name': definition['name'], 'parts': parts
            })

        return JsonResponse({'result': res})

    else:
        return HttpResponseBadRequest


