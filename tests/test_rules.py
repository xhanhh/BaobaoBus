from auto_answer.core.models import OCRBundle, OCRResult, Question
from auto_answer.solving.rules import RuleEngine


def question(text: str, options: tuple[str, str, str, str]) -> Question:
    empty = OCRResult("", 1.0, False)
    return Question(text, options, OCRBundle(empty, (empty, empty, empty, empty)))


def test_more_than_word_problem_matches_unique_option() -> None:
    result = RuleEngine().solve(question("比9多6的数是()", ("16", "15", "3", "10")))
    assert result is not None
    assert result.answer_index == 1


def test_basic_arithmetic() -> None:
    result = RuleEngine().solve(question("12/3+2=多少?", ("6", "4", "8", "5")))
    assert result is not None
    assert result.answer_index == 0


def test_currency_exchange() -> None:
    result = RuleEngine().solve(
        question("1张100元可以换()张10元.", ("20", "10", "100", "5"))
    )
    assert result is not None
    assert result.answer_index == 1


def test_remaining_word_problem() -> None:
    result = RuleEngine().solve(
        question("有15个气球,卖了9个,还剩下()个.", ("7", "4", "6", "5"))
    )
    assert result is not None
    assert result.answer_index == 2


def test_repeated_action_multiplies_per_action_by_times() -> None:
    result = RuleEngine().solve(
        question(
            "一袋糖有50块,每次拿8块,拿了4次,一共拿了()块.",
            ("24", "32", "40", "16"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_complete_groups_uses_floor_division() -> None:
    result = RuleEngine().solve(
        question(
            "有25个苹果,每6个装一袋,可以装满()袋.",
            ("5", "6", "4", "3"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_required_capacity_uses_ceiling_division() -> None:
    result = RuleEngine().solve(
        question(
            "有26个小朋友坐船,每条船坐5人,需要()条船.",
            ("5", "7", "6", "4"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_multiple_reductions_are_all_applied() -> None:
    result = RuleEngine().solve(
        question(
            "兔哥哥原来有8根胡萝卜,它先吃掉了2根,又吃掉了5根,"
            "请问兔哥哥现在还剩几根胡萝卜?",
            ("4", "2", "3", "1"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_mixed_inventory_actions_are_applied_in_order() -> None:
    result = RuleEngine().solve(
        question(
            "小华有24个气球,放飞了6个,又买了8个,现在有()个气球.",
            ("18", "26", "22", "30"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_total_reduction_answers_how_much_shorter() -> None:
    result = RuleEngine().solve(
        question(
            "一根绳子长15米,第一次用去4米,第二次用去5米,这根绳子短了()米.",
            ("9", "6", "10", "11"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_equivalent_option_expression_is_evaluated_locally() -> None:
    result = RuleEngine().solve(
        question(
            "与6+5结果相等的算式是().",
            ("2+9", "4+8", "6+6", "6+7"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_nearest_option_expression_below_threshold_is_selected() -> None:
    result = RuleEngine().solve(
        question(
            "下面()算式的得数比50小一些.",
            ("38+9", "42+9", "86-30", "53-2"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_unique_option_expression_below_threshold_is_selected() -> None:
    result = RuleEngine().solve(
        question(
            "得数小于4的算式是",
            ("2+1", "4+1", "5+0", "2+3"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_ambiguous_nearby_option_expressions_return_none() -> None:
    assert RuleEngine().solve(
        question(
            "下面()算式的得数比50小一些.",
            ("38+9", "48-1", "42+9", "53-2"),
        )
    ) is None


def test_square_blank_comparison_expression() -> None:
    result = RuleEngine().solve(
        question(
            "“13-7□5”,比较大小,在口里应填的符号是(",
            ("<", "=", "-", ">"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "comparison expression: 6>5"


def test_four_term_arithmetic_sequence_with_middle_blank() -> None:
    result = RuleEngine().solve(
        question(
            "照规律写数201、302、()、504,括号里的数是()",
            ("404", "304", "405", "403"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "arithmetic sequence: 403"


def test_unverified_sequence_returns_none() -> None:
    assert RuleEngine().solve(
        question(
            "照规律写数201、302、()、505,括号里的数是()",
            ("404", "304", "405", "403"),
        )
    ) is None


def test_queue_position_from_right() -> None:
    result = RuleEngine().solve(
        question(
            "有20个小朋友排队,从左数小强是第11个,从右数小强是第()个.",
            ("12", "9", "11", "10"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_younger_than_parent_age() -> None:
    result = RuleEngine().solve(
        question(
            "小明今年7岁,他比爸爸小25岁,爸爸今年()岁.",
            ("30", "18", "32", "24"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_unique_number_between_bounds() -> None:
    result = RuleEngine().solve(
        question("下列各数中,比55大,比60小的数是()", ("58", "55", "61", "60"))
    )
    assert result is not None
    assert result.answer_index == 0


def test_money_sum_with_units() -> None:
    result = RuleEngine().solve(
        question(
            "买一块橡皮4角,一支铅笔5角,一共需要().",
            ("1元", "9角", "9元", "2元"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_counted_money_sum_with_composite_unit_option() -> None:
    result = RuleEngine().solve(
        question(
            "小明有2张5角,3张1角,他共有()钱.",
            ("2元3角", "1元3角", "1元", "5元"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "counted money sum: 13角"


def test_counted_money_does_not_fall_back_to_face_value_sum() -> None:
    assert RuleEngine().solve(
        question(
            "小明有2张5角,3张1角,一共有()钱.",
            ("6角", "1元", "2元", "5元"),
        )
    ) is None


def test_number_between_neighbors() -> None:
    result = RuleEngine().solve(
        question("我的邻居是8和6,我是().", ("4", "9", "5", "7"))
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "number between neighbors: 7"


def test_nonconsecutive_neighbors_are_not_guessed() -> None:
    assert RuleEngine().solve(
        question("我的邻居是8和4,我是().", ("4", "9", "6", "7"))
    ) is None


def test_largest_two_digit_number_on_counter() -> None:
    result = RuleEngine().solve(
        question(
            "在计数器上,用5颗珠子可以表示的最大两位数是().",
            ("32", "41", "50", "23"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "counter 最大 two-digit number: 50"


def test_smallest_two_digit_number_on_counter() -> None:
    result = RuleEngine().solve(
        question(
            "在计数器上,用5颗珠子可以表示的最小两位数是().",
            ("50", "14", "23", "41"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "counter 最小 two-digit number: 14"


def test_impossible_two_digit_counter_bead_count_is_not_guessed() -> None:
    assert RuleEngine().solve(
        question(
            "在计数器上,用19颗珠子可以表示的最大两位数是().",
            ("99", "98", "90", "89"),
        )
    ) is None


def test_two_distinct_recipient_distribution_counts_ordered_allocations() -> None:
    result = RuleEngine().solve(
        question(
            "摘了7个桃子,分给两只小猴子,有几种分法?(每只小猴子最少分1个)()",
            ("3种", "6种", "2种", "5种"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "two distinct recipients: 6"


def test_group_photo_count_includes_the_named_child() -> None:
    result = RuleEngine().solve(
        question(
            "小红和小组里的每一个同学都合照一次像,一共照了9次."
            "小组里一共有多少人?",
            ("9人", "10人", "11人", "8人"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "photographer plus photographed classmates: 10"


def test_combined_total_when_second_person_has_fewer() -> None:
    result = RuleEngine().solve(
        question(
            "小明有15个苹果,小红的苹果比小明少7个,两人一共有()个苹果.",
            ("8", "23", "22", "20"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "word arithmetic: 23"


def test_largest_numeric_option() -> None:
    result = RuleEngine().solve(question("下面哪个数最大?", ("-1", "7", "3", "2")))
    assert result is not None
    assert result.answer_index == 1


def test_largest_value_fitting_inequality() -> None:
    result = RuleEngine().solve(
        question("在算式()+7<15中,括号里最大能填几?", ("10", "8", "7", "9"))
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "inequality maximum: 7"


def test_maximum_complete_groups_uses_floor_division() -> None:
    result = RuleEngine().solve(
        question(
            "做一个毽子要用4根羽毛,15根羽毛最多可以做()个毽子.",
            ("4", "3", "5", "2"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "word arithmetic: 3"


def test_contextual_maximum_does_not_fall_through_to_largest_option() -> None:
    assert RuleEngine().solve(
        question("每盒装4个,最多能装多少盒?", ("4", "3", "5", "2"))
    ) is None


def test_unsupported_inequality_returns_none_instead_of_largest_option() -> None:
    assert RuleEngine().solve(
        question("在复杂条件下括号里最大能填几?", ("10", "8", "7", "9"))
    ) is None


def test_ambiguous_rule_returns_none() -> None:
    assert RuleEngine().solve(question("选择正确答案", ("1", "2", "3", "4"))) is None


def test_duplicate_matching_options_returns_none() -> None:
    assert RuleEngine().solve(question("2+2=?", ("4", "4", "3", "5"))) is None
