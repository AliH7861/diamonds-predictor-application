import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.regression.api import start_prediction_api
from src.regression.feature_engineering import FEATURE_SETS
from src.regression.prediction import load_best_model, predict_prices, save_best_model
from src.regression.preprocessing import prepare_regression_data
from src.regression.training import evaluate_on_test, train_model
from tests.helpers import make_diamonds


EXAMPLE = {
    "carat": 0.70,
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS1",
    "depth": 61.5,
    "table": 57.0,
    "x": 5.70,
    "y": 5.72,
    "z": 3.51,
}


class RegressionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.data_path = Path(cls.temp.name) / "diamonds.csv"
        make_diamonds(rows=180).to_csv(cls.data_path, index=False)
        cls.prepared = prepare_regression_data(cls.data_path)
        cls.model_run = train_model(cls.prepared, "HUMAN_PLUS_RAW", "RandomForest", smoke=True)
        save_best_model(cls.model_run, cls.temp.name)
        cls.loaded = load_best_model(cls.temp.name)
        cls.server = start_prediction_api(cls.temp.name, port=0)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temp.cleanup()

    def test_cleaning_split_and_feature_dimensions(self):
        split_sizes = [len(self.prepared["raw"][name]) for name in ["train", "valid", "test"]]
        self.assertEqual(sum(split_sizes), len(self.prepared["featured"]))
        self.assertEqual(split_sizes, [126, 27, 27])
        expected = {"CURRENT_BASELINE": 34, "HUMAN_ONLY": 62, "HUMAN_PLUS_RAW": 67}
        for name, columns in expected.items():
            self.assertEqual(self.prepared["experiments"][name]["matrices"]["train"].shape[1], columns)
            self.assertTrue(np.isfinite(self.prepared["experiments"][name]["matrices"]["test"]).all())

    def test_price_is_target_and_not_a_feature(self):
        for config in FEATURE_SETS.values():
            self.assertNotIn("price", config["numeric"] + config["categorical"])
            self.assertIn("clarity", config["categorical"])

    def test_saved_pipeline_and_test_evaluation(self):
        before = predict_prices(self.model_run, EXAMPLE)
        after = predict_prices(self.loaded, EXAMPLE)
        self.assertEqual(before, after)
        predictions, metrics = evaluate_on_test(self.model_run, self.prepared)
        self.assertEqual(len(predictions), 27)
        self.assertGreater(metrics["MAE"], 0)

    def test_http_api_and_input_validation(self):
        request = Request(self.url + "/predict", data=json.dumps(EXAMPLE).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        self.assertEqual(result["predictions"], predict_prices(self.loaded, EXAMPLE))

        invalid = dict(EXAMPLE, price=2500)
        request = Request(self.url + "/predict", data=json.dumps(invalid).encode("utf-8"), method="POST")
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 400)
        error.exception.close()


if __name__ == "__main__":
    unittest.main()
