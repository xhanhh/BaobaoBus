from auto_answer.runtime.stats import SessionStats


def test_session_stats_count_rules_llms_fallbacks_and_results() -> None:
    stats = SessionStats()

    stats.record_question("rule")
    stats.record_question("aliyun-openai")
    stats.record_question("ollama")
    stats.record_question("random-fallback")
    stats.record_round(won=True)
    stats.record_round(won=False)

    assert stats.rounds == 2
    assert stats.questions == 4
    assert stats.rule_answers == 1
    assert stats.llm_answers == 2
    assert stats.fallback_answers == 1
    assert stats.victories == 1
    assert stats.failures == 1
    assert stats.as_log_message() == (
        "SESSION_STATS rounds=2 questions=4 rule=1 llm=2 "
        "fallback=1 victories=1 failures=1"
    )
