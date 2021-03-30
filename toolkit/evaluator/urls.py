from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register("evaluators", views.EvaluatorViewSet, basename="evaluator")
