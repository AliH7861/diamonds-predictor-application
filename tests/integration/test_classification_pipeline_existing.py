import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.classification import prepare_classification_data
from src.classification.api import start_prediction_api
from src.classification.datasets import get_dataset_frames
from src.classification.feature_engineering import engineer_features
from src.classification.prediction import save_best_model, load_best_model, predict_diamonds
from tests.helpers import make_diamonds


class ClassificationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.data_path = Path(cls.temp.name) / "diamonds.csv"
        make_diamonds(rows=180).to_csv(cls.data_path, index=False)
        cls.prepared = prepare_classification_data(cls.data_path)
        cls.example = {
            "carat": 0.7,
            "cut": "Ideal",
            "color": "G",
            "depth": 61.5,
            "table": 57,
            "x": 5.7,
            "y": 5.72,
            "z": 3.51,
            "price": 2500,
        }
        cls.models = {}
        for number in [1, 2]:
            data = cls.prepared["experiments"][number]
            model = GaussianNB().fit(data["X_train"], cls.prepared["y_train"])
            folder = Path(cls.temp.name) / str(number)
            save_best_model(
                model, "Tree", data["preprocessor"], number, "Verification model", {}, folder
            )
            cls.models[number] = (model, load_best_model(folder))
        cls.server = start_prediction_api(Path(cls.temp.name) / "2", port=0)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temp.cleanup()

    def test_saved_predictions_match_original_model(self):
        for number, (model, bundle) in self.models.items():
            features = engineer_features(pd.DataFrame([self.example]))
            X = (
                bundle["preprocessor"]
                .transform(features[bundle["metadata"]["features"]])
                .astype(np.float32)
            )
            expected = model.predict_proba(X)[0]
            result = predict_diamonds(bundle, self.example)[0]
            np.testing.assert_allclose(list(result["probabilities"].values()), expected)
            self.assertEqual(result["clarity_target"], int(model.predict(X)[0]))

    def test_price_requirement_tracks_winning_experiment(self):
        without_price = {key: value for key, value in self.example.items() if key != "price"}
        self.assertEqual(len(predict_diamonds(self.models[1][1], without_price)), 1)
        with self.assertRaisesRegex(ValueError, "price"):
            predict_diamonds(self.models[2][1], without_price)

    def test_http_predictions_and_validation(self):
        payload = [self.example, self.example]
        request = Request(
            self.url + "/predict",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        self.assertEqual(result["predictions"], predict_diamonds(self.models[2][1], payload))
        with urlopen(self.url + "/model", timeout=10) as response:
            self.assertIn("price", json.load(response)["required_inputs"])
        for bad in [dict(self.example, x=0), dict(self.example, clarity="IF"), {}]:
            request = Request(self.url + "/predict", data=json.dumps(bad).encode())
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=10)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()

    def test_exported_frames_match_model_inputs_and_labels(self):
        frames = get_dataset_frames(self.prepared)
        self.assertEqual(len(frames), 14)
        for number, data in self.prepared["experiments"].items():
            for split in ["train", "valid", "test"]:
                frame = frames[f"experiment_{number}_{split}_processed"]
                np.testing.assert_array_equal(
                    frame[data["preprocessor"].get_feature_names_out()], data[f"X_{split}"]
                )
                np.testing.assert_array_equal(frame["Clarity_Target"], self.prepared[f"y_{split}"])
                np.testing.assert_array_equal(
                    frame["Source_Row"],
                    self.prepared["diamonds"].index[self.prepared[f"{split}_indices"]],
                )
        self.assertEqual(set(frames["experiment_1"]["Split"]), {"train", "valid", "test"})


if __name__ == "__main__":
    unittest.main()
