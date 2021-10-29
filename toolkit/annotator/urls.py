from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register('annotator', views.AnnotatorViewset, basename='annotator')
router.register('labelset', views.LabelsetViewset, basename='labelset')