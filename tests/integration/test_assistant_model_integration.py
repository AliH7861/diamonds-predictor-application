"""Integration tests that execute the project's real saved ML artifacts."""

from pathlib import Path
from numbers import Integral

import pandas as pd
import pytest

from src.assistant.dataset_search import DiamondCatalog
from src.assistant.model_evidence import ModelEvidenceProvider
from src.assistant.routing import route_question
from src.assistant.service import DiamondAssistant
from src.assistant.tool_executor import execute_model_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIAMOND = {
    "carat": 1.0,
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2",
    "depth": 61.5,
    "table": 57.0,
    "x": 6.45,
    "y": 6.43,
    "z": 3.96,
}


class NoCallLLM:
    def complete(self, *_args, **_kwargs):
        raise AssertionError("Deterministic model inference must not call the LLM")

    def structured(self, *_args, **_kwargs):
        raise AssertionError("Deterministic model inference must not call the LLM")


class NoCallStores:
    def search_memory(self, *_args, **_kwargs):
        raise AssertionError("Deterministic model inference must not search memory")


@pytest.fixture(scope="module")
def provider():
    loaded = ModelEvidenceProvider(PROJECT_ROOT)
    assert loaded.status() == {
        "classification_loaded": True,
        "regression_loaded": True,
        "clustering_loaded": True,
        "errors": [],
    }
    return loaded


def test_real_classification_artifact_and_preprocessor_execute(provider):
    output, trace = provider.execute("classification", RAW_DIAMOND)
    assert output["predicted_class"] in {"I", "SI", "VS", "VVS", "IF"}
    assert set(output["class_probabilities"]) == {"I", "SI", "VS", "VVS", "IF"}
    assert trace["inference_success"] is True
    assert Path(trace["preprocessor"]).is_file()


def test_real_regression_artifact_and_preprocessor_execute(provider):
    output, trace = provider.execute("regression", RAW_DIAMOND)
    assert output["prediction"] > 0
    assert output["source"] == "regression_model"
    assert trace["inference_success"] is True
    assert Path(trace["preprocessor"]).is_file()


def test_real_clustering_artifact_scaler_and_profile_execute(provider):
    output, trace = provider.execute("clustering", {**RAW_DIAMOND, "price": 6200})
    assert isinstance(output["cluster_id"], int)
    assert output["cluster_profile"]["name"]
    assert "not validated customer psychology" in output["interpretation_limit"]
    assert trace["inference_success"] is True


def test_multi_tool_plan_executes_real_models_in_dependency_order(provider):
    evidence = execute_model_plan(
        provider,
        ["classification", "regression", "clustering"],
        RAW_DIAMOND,
    )
    assert evidence["tools_executed"] == ["classification", "regression", "clustering"]
    assert evidence["tools_skipped"] == []
    assert evidence["tool_trace"][2]["resolved_dependencies"] == {"price": "regression.prediction"}
    assert evidence["answer_sources"] == {
        "predicted_clarity": "classification_model",
        "predicted_price": "regression_model",
        "cluster_profile": "clustering_model",
    }


def test_missing_features_skip_only_affected_models(provider):
    partial = {key: value for key, value in RAW_DIAMOND.items() if key != "clarity"}
    evidence = execute_model_plan(provider, ["classification", "regression"], partial)
    assert evidence["tools_executed"] == ["classification"]
    assert evidence["tools_skipped"][0]["tool"] == "regression"
    assert evidence["tools_skipped"][0]["missing_features"] == ["clarity"]


def test_router_preserves_three_model_request():
    route = route_question(
        "Predict clarity, estimate the price, and tell me what profile this diamond fits."
    )
    assert route.intent == "model_prediction"
    assert route.level_3 == "MULTI_MODEL"
    assert all(name in route.reason for name in ("classification", "regression", "clustering"))


def test_assistant_runs_real_multi_model_path_without_llm(provider):
    catalog_row = {**RAW_DIAMOND, "price": 6200}
    assistant = DiamondAssistant(
        NoCallLLM(),
        DiamondCatalog(pd.DataFrame([catalog_row])),
        NoCallStores(),
        enricher=provider,
    )
    question = (
        "Predict clarity, estimate price, and tell me what profile this diamond fits: "
        "1.0 carat Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96."
    )
    result = assistant.ask(question)
    assert result["status"] == "answered"
    assert result["evidence"]["tools_executed"] == [
        "classification",
        "regression",
        "clustering",
    ]
    assert result["evidence"]["rag_context"] == []
    assert result["trace"][-1]["stage"] == "final_answer"
    assert result["trace"][-1]["result"] == result["answer"]


def test_real_models_can_enrich_actual_dataframe_rows(provider):
    rows = pd.DataFrame([{**RAW_DIAMOND, "price": 6200}])
    enriched = provider.enrich(rows)
    assert enriched.loc[0, "model_price"] > 0
    assert enriched.loc[0, "predicted_clarity_family"] in {"I", "SI", "VS", "VVS", "IF"}
    assert isinstance(enriched.loc[0, "buyer_segment"], Integral)


def test_model_and_dataframe_comparison_preserves_both_sources(provider):
    rows = pd.DataFrame(
        [
            {**RAW_DIAMOND, "price": 6200},
            {**RAW_DIAMOND, "price": 6400, "carat": 1.02, "x": 6.5, "y": 6.48},
        ]
    )
    assistant = DiamondAssistant(
        NoCallLLM(), DiamondCatalog(rows), NoCallStores(), enricher=provider
    )
    result = assistant.ask(
        "Predict the price and compare it with actual dataset values: 1.0 carat "
        "Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96."
    )
    assert result["evidence"]["tools_executed"] == ["regression", "dataframe"]
    assert result["evidence"]["answer_sources"]["predicted_price"] == "regression_model"
    assert result["evidence"]["answer_sources"]["observed_price_summary"] == "dataframe"
    assert result["evidence"]["dataset_stats"]["observed_price_median"] == 6300
