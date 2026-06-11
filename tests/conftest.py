import sys
from unittest.mock import MagicMock

# consumer.py instantiates PersonCNNModel() and PersonRecognizer() at module scope,
# which would fail without the actual .keras and .tflite artefacts on disk.
# Inject lightweight mocks before pytest collects any test module that imports consumer.
# test_person_model_loader.py and test_inference_person.py pop these entries and
# load the real modules directly.

_person_loader_mod = MagicMock(name="person_model_loader")
_person_loader_mod.PersonCNNModel = MagicMock(return_value=MagicMock())
sys.modules["app.model.person.person_model_loader"] = _person_loader_mod

_inference_mod = MagicMock(name="inference_person")
_inference_mod.PersonRecognizer = MagicMock(return_value=MagicMock())
sys.modules["app.model.person.inference_person"] = _inference_mod
