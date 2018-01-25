from django.urls import path

from . import views

urlpatterns = [
    path('map/<slug:condition>/<slug:feedback>', views.index, name='index'),
    path('training', views.training, name='training'),
    path('deictic_models', views.deictic_models, name='deictic_models'),
    path('deictic_eval', views.deictic_eval, name='deictic_eval')
]
