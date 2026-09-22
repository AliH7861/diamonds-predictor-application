from src.assistant.domain_answers import answer_domain_catalog


def test_cut_catalog_answer_is_short_and_uses_the_five_dataset_grades():
    answer = answer_domain_catalog("cut", [])

    assert answer is not None
    assert all(grade in answer for grade in ("Fair", "Good", "Very Good", "Premium", "Ideal"))
    assert len(answer.split()) < 50
