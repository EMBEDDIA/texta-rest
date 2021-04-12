from rest_framework import routers
from .views import SummarizerViewSet

router = routers.DefaultRouter()
router.register('summarizer', SummarizerViewSet, basename='summarize')