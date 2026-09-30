from jspace.quality import max_ngram_repetition


def test_repetition_metric_handles_short_and_repeated_sequences():
    assert max_ngram_repetition([1, 2, 3]) == 0.0
    repeated = [1, 2, 3, 4] * 10
    assert max_ngram_repetition(repeated) > 0.5
