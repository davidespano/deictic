from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('deictic_models', views.deictic_models, name='deictic_models')
]