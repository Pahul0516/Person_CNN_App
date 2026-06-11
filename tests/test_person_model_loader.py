import sys
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Remove the conftest stub so this file loads the real module.
sys.modules.pop("app.model.person.person_model_loader", None)

from app.model.person.person_model_loader import PersonCNNModel


class TestPersonCNNModel:

    @patch("app.model.person.person_model_loader.os.path.exists", return_value=False)
    def test_raises_when_model_file_missing(self, _):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            PersonCNNModel("/fake/model.keras", "/fake/classes.json")

    @patch("app.model.person.person_model_loader.tf")
    @patch("app.model.person.person_model_loader.os.path.exists")
    def test_raises_when_json_file_missing(self, mock_exists, mock_tf):
        # Model path exists (.keras), JSON path does not (.json)
        mock_exists.side_effect = lambda p: p.endswith(".keras")

        with pytest.raises(FileNotFoundError, match="JSON labels not found"):
            PersonCNNModel("/fake/model.keras", "/fake/classes.json")

    @patch("app.model.person.person_model_loader.tf")
    @patch("app.model.person.person_model_loader.os.path.exists", return_value=True)
    def test_stores_loaded_model(self, _mock_exists, mock_tf):
        mock_model = MagicMock()
        mock_tf.keras.models.load_model.return_value = mock_model
        classes = {"class_names": ["Alice", "Bob"]}

        with patch("builtins.open", mock_open(read_data=json.dumps(classes))):
            loader = PersonCNNModel("/some/model.keras", "/some/classes.json")

        mock_tf.keras.models.load_model.assert_called_once_with("/some/model.keras")
        assert loader.model is mock_model

    @patch("app.model.person.person_model_loader.tf")
    @patch("app.model.person.person_model_loader.os.path.exists", return_value=True)
    def test_stores_class_names_from_json(self, _mock_exists, mock_tf):
        categories = ["Alice", "Bob", "Charlie"]
        classes = {"class_names": categories}

        with patch("builtins.open", mock_open(read_data=json.dumps(classes))):
            loader = PersonCNNModel("/some/model.keras", "/some/classes.json")

        assert loader.categories == categories
