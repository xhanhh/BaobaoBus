from auto_answer.core.models import OCRBundle, OCRResult, Question
from auto_answer.solving.rules import RuleEngine


def question(text: str, options: tuple[str, str, str, str]) -> Question:
    empty = OCRResult("", 1.0, False)
    return Question(text, options, OCRBundle(empty, (empty, empty, empty, empty)))


def test_high_confidence_templates_from_aliyun_run_use_local_rules() -> None:
    cases = (
        (
            "一壶水可以装满5个水瓶,一个水瓶可以装满2杯水,"
            "一壶水可以装满()杯水.",
            ("3", "8", "10", "7"),
            2,
        ),
        (
            "一个两位数,十位上的数字比个位上的数字小4,这个数可能是().",
            ("14", "40", "15", "51"),
            2,
        ),
        ("分针走一圈,时针走()大格.", ("12", "5", "1", "60"), 2),
        (
            "小刚有12张贴纸,送给小红4张,两人就一样多,"
            "小红原来有()张贴纸.",
            ("6", "4", "10", "8"),
            1,
        ),
        ("一个物体都没有用()表示.", ("1", "0.1", "10", "0"), 3),
        ("减数是7,被减数是15,差是().", ("9", "7", "22", "8"), 3),
        (
            "比最小的两位数多7的数是().",
            ("3", "17", "70", "7"),
            1,
        ),
        (
            "爸爸今年36岁,小明今年8岁,10年后爸爸比小明大()岁.",
            ("18", "38", "46", "28"),
            3,
        ),
        (
            "小猴有15个香蕉,第一天吃了5个,第二天吃了4个,"
            "第三天吃了()个就正好吃完.",
            ("9", "6", "5", "4"),
            1,
        ),
        (
            "用两个完全一样的正方形可以拼成一个().",
            ("三角形", "正方形", "圆形", "长方形"),
            3,
        ),
        (
            "有12个小朋友要坐船,一条船最多坐4人,他们至少需要()条船.",
            ("5", "3", "4", "2"),
            1,
        ),
        (
            "小明做了10道口算题,小芳比他少做了3道,"
            "小芳做了()道.",
            ("13", "7", "8", "6"),
            1,
        ),
        (
            "妈妈买了24个桔子,6个装一袋,可以装()袋.",
            ("6", "3", "5", "4"),
            3,
        ),
        ("8连续加3次,和是().", ("11", "32", "16", "24"), 1),
        (
            "一个数,个位上是0,十位上是3,这个数是(),"
            "它后面第5个数是().",
            ("3和8", "30和35", "3和5", "30和25"),
            1,
        ),
        (
            "有红、白、蓝三种颜色的气球共18个,其中红气球有5个,"
            "白气球有8个,蓝气球有()个.",
            ("3", "4", "6", "5"),
            3,
        ),
        (
            "小明的储蓄罐里有5角和1元的硬币共10元,"
            "其中5角硬币有6枚,1元硬币有()枚.",
            ("8", "6", "4", "7"),
            3,
        ),
        ("时针从“2”走到“5”,走了()小时.", ("5", "3", "4", "2"), 1),
        ("现在是上午9时,2小时前是()时.", ("10", "8", "11", "7"), 3),
        (
            "公交车到站后,下去10人,上来7人."
            "这时车上的人数与原来相比,()了()人.",
            ("少,17", "多,3", "多,17", "少,3"),
            3,
        ),
        (
            "小芳有14支彩笔,小丽的彩笔比小芳少5支,"
            "小丽有()支彩笔.",
            ("9", "10", "19", "8"),
            0,
        ),
        (
            "一箱牛奶有12盒,喝掉一半后还剩()盒.",
            ("24", "6", "0", "12"),
            1,
        ),
        (
            "学校合唱队有男生15人,女生比男生多4人,女生有()人.",
            ("11", "19", "9", "20"),
            1,
        ),
        (
            "3个小朋友下棋,每两个人都要下一盘,一共要下()盘.",
            ("3", "6", "2", "4"),
            0,
        ),
        (
            "小明家养了8只白兔,黑兔比白兔少2只,"
            "小明家一共养了()只兔.",
            ("16", "10", "6", "14"),
            3,
        ),
    )

    for text, options, expected_index in cases:
        result = RuleEngine().solve(question(text, options))
        assert result is not None, text
        assert result.source == "rule", text
        assert result.answer_index == expected_index, text


def test_digit_occurrences_in_inclusive_range() -> None:
    result = RuleEngine().solve(
        question(
            "小华从1写到50,一共写了()个数字“5”.",
            ("10", "5", "6", "1"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "digit occurrences in inclusive range: 6"


def test_zero_occurrences_do_not_count_leading_zeroes() -> None:
    result = RuleEngine().solve(
        question(
            "小华从1写到100,一共写了()个数字“0”.",
            ("10", "11", "20", "1"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_transfer_with_nonzero_remaining_difference() -> None:
    result = RuleEngine().solve(
        question(
            "哥哥有15本书,弟弟有9本书,哥哥给弟弟()本,"
            "哥哥还比弟弟多2本.",
            ("4", "2", "6", "3"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "word arithmetic: 2"


def test_transfer_with_impossible_fractional_count_returns_none() -> None:
    result = RuleEngine().solve(
        question(
            "哥哥有14本书,弟弟有9本书,哥哥给弟弟()本,"
            "哥哥还比弟弟多2本.",
            ("1", "2", "3", "4"),
        )
    )
    assert result is None


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


def test_broken_then_bought_inventory_applies_both_actions() -> None:
    result = RuleEngine().solve(
        question(
            "小明有14个气球,破了6个,又买了5个,现在有()个气球.",
            ("13", "19", "9", "8"),
        )
    )
    assert result is not None
    assert result.answer_index == 0
    assert result.reason == "word arithmetic: 13"


def test_inventory_does_not_ignore_unknown_numeric_action() -> None:
    result = RuleEngine().solve(
        question(
            "小明有14个气球,神秘变化了6个,又买了5个,现在有()个气球.",
            ("13", "19", "9", "8"),
        )
    )
    assert result is None


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


def test_multi_step_inventory_with_picking_and_eating() -> None:
    result = RuleEngine().solve(
        question(
            "小猴摘了14个桃子,吃了6个,又摘了3个,现在有()个桃子.",
            ("17", "20", "8", "11"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "word arithmetic: 11"


def test_remaining_story_selects_semantically_correct_equation() -> None:
    result = RuleEngine().solve(
        question(
            "妈妈买回来12个鸡蛋,做饭用了4个,还剩几个?正确的算式是().",
            ("12+4=16", "4+8=12", "12-8=4", "12-4=8"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "remaining quantity equation: 12-4=8"


def test_people_behind_excludes_the_named_child() -> None:
    result = RuleEngine().solve(
        question(
            "16个小朋友排队,小明前面有7人,小明后面有()人.",
            ("10", "7", "8", "9"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "word arithmetic: 8"


def test_queue_position_accepts_pai_cheng_yi_pai_wording() -> None:
    result = RuleEngine().solve(
        question(
            "20个小朋友排成一排,从左数小红排第6,从右数小红排第().",
            ("16", "14", "5", "15"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "word arithmetic: 15"


def test_largest_digit_constructs_two_digit_number() -> None:
    result = RuleEngine().solve(
        question(
            "一个两位数,十位上的数字是最大的一位数,"
            "个位上的数字比十位上的数字少2,这个数是().",
            ("79", "77", "99", "97"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "two-digit place value: 97"


def test_fixed_tens_digit_and_ones_addition_construct_number() -> None:
    result = RuleEngine().solve(
        question(
            "一个两位数,十位上的数字是5,"
            "个位上的数字是十位上的数字加上3,这个数是().",
            ("85", "35", "53", "58"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "two-digit place value: 58"


def test_place_value_accepts_xiao_as_less_than() -> None:
    result = RuleEngine().solve(
        question(
            "一个两位数,十位上的数字是6,"
            "个位上的数字比十位上的数字小4,这个数是().",
            ("62", "26", "46", "64"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_counts_integers_strictly_between_bounds() -> None:
    result = RuleEngine().solve(
        question("比8大,比11小的数有()个.", ("2", "4", "1", "3"))
    )
    assert result is not None
    assert result.answer_index == 0
    assert result.reason == "counting: 2"


def test_counts_pages_inclusively() -> None:
    result = RuleEngine().solve(
        question(
            "小明今天从第10页读到第15页,他今天读了()页.",
            ("7", "4", "6", "5"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "counting: 6"


def test_queue_total_from_people_in_front_and_behind() -> None:
    result = RuleEngine().solve(
        question(
            "同学们排队做操,我的前面有6人,后面有12人,这一排一共有几人?",
            ("20", "17", "19", "18"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "counting: 19"


def test_people_between_two_queue_positions() -> None:
    result = RuleEngine().solve(
        question(
            "小朋友排队,小明排第9,小红排第15,他们之间有()人.",
            ("8", "7", "5", "6"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "counting: 5"


def test_queue_total_from_positions_at_both_ends() -> None:
    result = RuleEngine().solve(
        question(
            "小朋友们排队,从前面数小亮是第10个,"
            "从后面数他是第5个,这一队共有()人.",
            ("16", "15", "14", "13"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "counting: 14"


def test_equivalent_expression_accepts_optional_de() -> None:
    result = RuleEngine().solve(
        question(
            "与4+2的结果相等的算式是().",
            ("3+2", "3+3", "1+2", "2+3"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_eagle_game_excludes_eagle_and_hen() -> None:
    result = RuleEngine().solve(
        question(
            "9个小朋友玩老鹰捉小鸡,已经捉到了4只小鸡,"
            "还有()只小鸡没被捉到.",
            ("6", "3", "4", "5"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "word arithmetic: 3"


def test_money_change_with_yuan_and_jiao() -> None:
    result = RuleEngine().solve(
        question(
            "一本故事书的价格是5元8角,付出一张10元,应找回().",
            ("4元8角", "5元8角", "5元2角", "4元2角"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "money change: 42角"


def test_money_change_with_bare_yuan_options() -> None:
    result = RuleEngine().solve(
        question(
            "一个书包35元,妈妈付了50元,应找回()元.",
            ("20", "15", "25", "85"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_power_outage_means_no_lights_are_lit() -> None:
    result = RuleEngine().solve(
        question(
            "教室里有20盏灯,全部亮着,突然停电了,关了9盏灯,"
            "教室里还有()盏灯亮着.",
            ("11", "20", "0", "9"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "word arithmetic: 0"


def test_equal_transfer_between_two_people() -> None:
    result = RuleEngine().solve(
        question(
            "小美有8支铅笔,小刚有4支铅笔,"
            "小美给小刚()支后,两人的铅笔就一样多.",
            ("3", "4", "2", "1"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "word arithmetic: 2"


def test_equal_transfer_accepts_made_items_wording() -> None:
    result = RuleEngine().solve(
        question(
            "小丽做了10朵花,小芳做了8朵花,"
            "小丽给小芳()朵,两人的花就一样多.",
            ("3", "4", "1", "2"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_trailing_increasing_arithmetic_sequence() -> None:
    result = RuleEngine().solve(
        question("找规律:2,4,6,8,().", ("10", "12", "7", "9"))
    )
    assert result is not None
    assert result.answer_index == 0


def test_trailing_decreasing_arithmetic_sequence() -> None:
    result = RuleEngine().solve(
        question("找规律:19,17,15,13,().", ("11", "10", "12", "9"))
    )
    assert result is not None
    assert result.answer_index == 0


def test_ten_tens_is_one_hundred() -> None:
    result = RuleEngine().solve(
        question("10个十是().", ("100", "1", "1000", "10"))
    )
    assert result is not None
    assert result.answer_index == 0


def test_reverse_place_value_relation() -> None:
    result = RuleEngine().solve(
        question(
            "一个数个位上是3,十位上的数比个位上的数多2,这个数是().",
            ("23", "32", "35", "53"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_reverse_more_than_blank() -> None:
    result = RuleEngine().solve(
        question("10比()多4.", ("8", "7", "6", "14"))
    )
    assert result is not None
    assert result.answer_index == 2


def test_yuan_to_fen_conversion() -> None:
    result = RuleEngine().solve(
        question("1元=()分.", ("10", "100", "60", "1000"))
    )
    assert result is not None
    assert result.answer_index == 1


def test_clock_face_at_exact_hour() -> None:
    result = RuleEngine().solve(
        question(
            "下面钟面上是6时的是().",
            (
                "时针指着12,分针指着6",
                "时针指着6,分针指着6",
                "时针指着12,分针指着12",
                "时针指着6,分针指着12",
            ),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_elapsed_whole_hours() -> None:
    result = RuleEngine().solve(
        question(
            "我早上8时上学,中午12时吃饭,我上学后()小时吃饭.",
            ("3", "6", "5", "4"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_unique_value_between_reverse_qualitative_bounds() -> None:
    result = RuleEngine().solve(
        question(
            "一个数比40少一些,比30多一些,这个数可能是().",
            ("45", "29", "35", "25"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "unique value between bounds"


def test_mixed_note_exchange_validates_both_counts() -> None:
    result = RuleEngine().solve(
        question(
            "一张50元可以换()张20元和()张10元.",
            (
                "3张20元和0张10元",
                "1张20元和5张10元",
                "2张20元和3张10元",
                "2张20元和1张10元",
            ),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "mixed note exchange: 50元"


def test_inventory_used_then_bought_does_not_skip_used_action() -> None:
    result = RuleEngine().solve(
        question(
            "小刚有12本练习本,用了5本,又买了4本,现在有()本.",
            ("7", "13", "16", "11"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "word arithmetic: 11"


def test_two_equal_addends() -> None:
    result = RuleEngine().solve(
        question("两个加数都是8,和是().", ("16", "64", "8", "0"))
    )
    assert result is not None
    assert result.answer_index == 0


def test_additive_identity_element_keeps_original_number() -> None:
    result = RuleEngine().solve(
        question(
            "一个数加(),和还是原来的数.",
            ("100", "0", "10", "1"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "identity element: 0"


def test_total_consumed_over_two_days() -> None:
    result = RuleEngine().solve(
        question(
            "妈妈买了20个苹果,第一天吃了6个,第二天吃了7个,"
            "两天一共吃了()个.",
            ("1", "12", "13", "14"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_total_that_flew_away() -> None:
    result = RuleEngine().solve(
        question(
            "树上有15只鸟,先飞走了6只,又飞走了3只,树上少了()只鸟.",
            ("12", "6", "3", "9"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_box_count_times_items_per_box() -> None:
    result = RuleEngine().solve(
        question(
            "有3盒巧克力,每盒10块,一共有()块巧克力.",
            ("10", "13", "33", "30"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_two_category_total() -> None:
    result = RuleEngine().solve(
        question(
            "一年级有男生18人,女生20人,一年级一共有()人.",
            ("22", "38", "40", "2"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_option_expression_maximum() -> None:
    result = RuleEngine().solve(
        question(
            "在12-4、11-5、14-9、13-7这些算式中,得数最大的是().",
            ("11-5", "14-9", "13-7", "12-4"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_same_tens_and_ones_digit_with_bound() -> None:
    result = RuleEngine().solve(
        question(
            "一个数,它的个位和十位上的数字相同,且比50大,这个数可能是().",
            ("22", "55", "44", "33"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_unique_even_number_between_bounds() -> None:
    result = RuleEngine().solve(
        question(
            "一个数比67大,比70小,并且是偶数,这个数是().",
            ("67", "69", "68", "70"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_equal_transfer_accepts_equal_wording_and_money_units() -> None:
    result = RuleEngine().solve(
        question(
            "哥哥有15元钱,弟弟有9元钱,"
            "哥哥给弟弟()元后,两人的钱数就相等.",
            ("4", "6", "5", "3"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_bus_inventory_accepts_passenger_wording() -> None:
    result = RuleEngine().solve(
        question(
            "公交车上有乘客25人,到站下车10人,上车8人,"
            "现在车上有()人.",
            ("33", "23", "27", "15"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "word arithmetic: 23"


def test_future_age_of_older_sibling() -> None:
    result = RuleEngine().solve(
        question(
            "我今年6岁,姐姐比我大4岁,5年后姐姐()岁.",
            ("10", "15", "11", "14"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "word arithmetic: 15"


def test_opposite_addend_changes_leave_sum_unchanged() -> None:
    result = RuleEngine().solve(
        question(
            "一个加数增加5,另一个加数减少5,和().",
            ("减少10", "不变", "增加5", "增加10"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "operation change: 不变"


def test_minuend_up_and_subtrahend_down_increase_difference() -> None:
    result = RuleEngine().solve(
        question(
            "被减数增加6,减数减少6,差().",
            ("减少12", "增加12", "不变", "增加6"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "operation change: 增加12"


def test_queue_total_accepts_this_row_wording() -> None:
    result = RuleEngine().solve(
        question(
            "小朋友们排队做操,从左数小刚排第5,"
            "从右数小刚排第6,这一排一共有()人.",
            ("10", "9", "12", "11"),
        )
    )
    assert result is not None
    assert result.answer_index == 0
    assert result.reason == "counting: 10"


def test_equal_transfer_when_second_person_is_giver() -> None:
    result = RuleEngine().solve(
        question(
            "小丽有7颗糖,小东有11颗糖,"
            "小东给小丽()颗后,两人的糖就一样多.",
            ("3", "2", "4", "1"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "word arithmetic: 2"


def test_analog_clock_between_hours() -> None:
    result = RuleEngine().solve(
        question(
            "钟面上,时针指在5和6之间,分针指着11,这时大约是().",
            ("05:11:00", "06:05:00", "06:55:00", "05:55:00"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "clock hands: 05:55"


def test_two_and_half_boxes() -> None:
    result = RuleEngine().solve(
        question(
            "一盒铅笔有10支,两盒半这样的铅笔一共有()支.",
            ("25", "30", "20", "15"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_bought_bags_times_items_per_bag() -> None:
    result = RuleEngine().solve(
        question(
            "妈妈买了3袋苹果,每袋6个,一共买了()个苹果.",
            ("18", "9", "12", "15"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_right_side_count_from_total_and_left_side() -> None:
    result = RuleEngine().solve(
        question(
            "13个同学排队,小华左边有4人,他右边有()人.",
            ("9", "8", "7", "10"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_minimum_ten_yuan_notes_for_two_items() -> None:
    result = RuleEngine().solve(
        question(
            "一个文具盒8元,一本笔记本3元,"
            "买这两样东西,至少要付()张10元.",
            ("1", "3", "4", "2"),
        )
    )
    assert result is not None
    assert result.answer_index == 3
    assert result.reason == "minimum notes required: 2"


def test_cages_required_with_items_first_wording() -> None:
    result = RuleEngine().solve(
        question(
            "有16只小鸡,每4只装一个笼子,需要()个笼子.",
            ("5", "4", "6", "3"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_sum_of_largest_one_digit_and_smallest_two_digit() -> None:
    result = RuleEngine().solve(
        question(
            "最大的一位数和最小的两位数的和是().",
            ("19", "9", "20", "10"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_subtrahend_from_minuend_and_difference() -> None:
    result = RuleEngine().solve(
        question(
            "被减数是14,差是6,减数是().",
            ("9", "20", "8", "7"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_subtrahend_when_difference_is_stated_first() -> None:
    result = RuleEngine().solve(
        question(
            "两个数的差是5,被减数是12,减数是().",
            ("7", "6", "8", "17"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_repeated_subtraction_count_to_zero() -> None:
    result = RuleEngine().solve(
        question(
            "从14里连续减去2,减()次后结果是0.",
            ("6", "5", "7", "8"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_inventory_moved_out_then_back_in() -> None:
    result = RuleEngine().solve(
        question(
            "教室里有20张桌子,搬走6张,又搬来4张,"
            "现在有()张桌子.",
            ("22", "14", "18", "10"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_place_value_accepts_short_tens_wording() -> None:
    result = RuleEngine().solve(
        question(
            "一个数,十位上是1,个位上的数字比十位上的数字多8,"
            "这个数是().",
            ("18", "19", "81", "9"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_same_amount_in_morning_and_afternoon() -> None:
    result = RuleEngine().solve(
        question(
            "学校图书馆上午借出9本书,下午借出的和上午同样多,"
            "图书馆一天借出()本书.",
            ("18", "9", "27", "0"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_same_pages_yesterday_and_today() -> None:
    result = RuleEngine().solve(
        question(
            "小明昨天看了6页书,今天看的页数和昨天一样多,"
            "他两天一共看了()页.",
            ("8", "12", "10", "6"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_trip_water_supply_is_not_enough() -> None:
    result = RuleEngine().solve(
        question(
            "有43个学生和3个老师去春游,每人一瓶水,"
            "准备45瓶水,够分配么?",
            ("无法确定", "不够", "够", "正好"),
        )
    )
    assert result is not None
    assert result.answer_index == 1
    assert result.reason == "supplies comparison: 45 vs 46"


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


def test_age_difference_between_siblings() -> None:
    result = RuleEngine().solve(
        question(
            "我今年6岁,姐姐15岁,姐姐比我大几岁?",
            ("6", "8", "7", "9"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_quantity_times_unit_price_then_change() -> None:
    result = RuleEngine().solve(
        question(
            "妈妈买了3斤苹果,每斤4元,付了20元,应找回()元.",
            ("8", "16", "12", "7"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_clock_hands_overlap_at_whole_hour() -> None:
    result = RuleEngine().solve(
        question(
            "()时整,分针和时针重合在一起.",
            ("6", "12", "3", "9"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_weight_inventory_with_consumption_and_restock() -> None:
    result = RuleEngine().solve(
        question(
            "一袋大米重10千克,吃了3千克,又买来4千克,现在重()千克.",
            ("11", "7", "14", "13"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_available_money_minus_single_item_price() -> None:
    result = RuleEngine().solve(
        question(
            "小华有30元钱,买了一个书包25元,应找回()元.",
            ("5", "10", "15", "55"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_people_behind_accepts_ge_ren_wording() -> None:
    result = RuleEngine().solve(
        question(
            "10个同学排队做操,小明前面有8个人,后面有()个人.",
            ("3", "1", "2", "0"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_direct_ones_and_tens_digits() -> None:
    result = RuleEngine().solve(
        question(
            "个位上是7,十位上是1,这个数是().",
            ("71", "17", "6", "8"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_consecutive_natural_numbers_are_solved_before_generic_extreme() -> None:
    result = RuleEngine().solve(
        question(
            "有10个连续的自然数,它们的和是45,其中最大的数是().",
            ("7", "10", "9", "8"),
        )
    )
    assert result is not None
    assert result.answer_index == 2
    assert result.reason == "consecutive natural numbers: 9"


def test_minute_hand_large_grid_to_second_hand_rotations() -> None:
    result = RuleEngine().solve(
        question(
            "分针走1大格,秒针走()圈.",
            ("10", "1", "5", "60"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_minimum_additional_amount_to_exceed() -> None:
    result = RuleEngine().solve(
        question(
            "小丽做了14朵花,小明做了6朵.小明至少还要做()朵,才能超过小丽?",
            ("10朵", "8朵", "7朵", "9朵"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_option_expression_result_in_seventies() -> None:
    result = RuleEngine().solve(
        question(
            "哪道题的得数是七十多?()",
            ("74+8", "63+6", "73+7", "64+8"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_mixed_yuan_jiao_subtraction() -> None:
    result = RuleEngine().solve(
        question(
            "1元-4角=()角.",
            ("5", "4", "10", "6"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_maximum_two_digit_number_with_digit_sum() -> None:
    result = RuleEngine().solve(
        question(
            "一个两位数,个位和十位上的数字和是8,这个数最大是().",
            ("71", "62", "88", "80"),
        )
    )
    assert result is not None
    assert result.answer_index == 3


def test_container_weight_after_half_contents_used() -> None:
    result = RuleEngine().solve(
        question(
            "一桶油连桶重12千克,用去一半油后,连桶重7千克,桶重()千克.",
            ("2", "10", "4", "5"),
        )
    )
    assert result is not None
    assert result.answer_index == 0


def test_count_of_ones_making_ten() -> None:
    result = RuleEngine().solve(
        question(
            "一个一个地数,()个一是10.",
            ("1", "10", "100", "20"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_movie_end_time_with_hours_and_minutes() -> None:
    result = RuleEngine().solve(
        question(
            "今晚电影7时开始,电影时长1小时30分,结束时间是().",
            ("8时", "08:30:00", "9时", "07:30:00"),
        )
    )
    assert result is not None
    assert result.answer_index == 1


def test_two_basket_remainder() -> None:
    result = RuleEngine().solve(
        question(
            "有14个苹果,放在两个篮子里,一个篮子放8个,另一个篮子放()个.",
            ("8", "10", "6", "22"),
        )
    )
    assert result is not None
    assert result.answer_index == 2


def test_contextual_largest_number_does_not_select_largest_option() -> None:
    assert RuleEngine().solve(
        question(
            "若干个连续自然数的和已知,其中最大的数是().",
            ("7", "10", "9", "8"),
        )
    ) is None


def test_high_value_templates_from_1918_run_are_local() -> None:
    cases = (
        (
            "3*9表示().",
            ("9个3相乘", "9和3相加", "3和9相加", "9个3相加"),
            3,
        ),
        (
            "与4+4+4表示的意思不相同的是().",
            ("4*3", "3个4相加", "4+3", "3*4"),
            2,
        ),
        (
            "下列算式中能用乘法算式表示的是().",
            ("5+5+5", "5+5+4", "5+4+3", "4+4+4+3"),
            0,
        ),
        (
            "下面的哪个算式可以用口诀“三五十五”来口算().",
            ("3+3+5", "3+3+3", "5+5+3", "5*3"),
            3,
        ),
        (
            "与算式2+2+2的结果不相等的是().",
            ("3*3", "3+3", "3*2", "2*3"),
            0,
        ),
        (
            "下面()算式不可以用“2*3”表示.",
            ("2+3", "3+3", "3*2", "2+2+2"),
            0,
        ),
        (
            "4个3相加的和加上2个3相加的和,与()的结果相等.",
            ("5*4", "6*3", "7*3", "5*3"),
            1,
        ),
        (
            "3+3+3+3+4可以改写成().",
            ("3*4+1", "3*5-1", "3*4", "3*5+1"),
            3,
        ),
        (
            "7+7+7+7+5+2+1可以改写成().",
            ("7*5", "7*5+1", "7*4", "7*5+3"),
            1,
        ),
        (
            "求7个2相加的和是多少?正确的列式是().",
            ("7+7", "7+2", "7*2", "7*7"),
            2,
        ),
        (
            "如果两个因数都是9,那么积是().",
            ("18", "99", "81", "9"),
            2,
        ),
        (
            "在5*8的算式中,给第一个乘数增加1,另一个乘数减少1,积会()",
            ("不能确定", "变大", "不变", "变小"),
            1,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text
        assert result.source == "rule", text


def test_combinatorics_templates_from_1918_run_are_local() -> None:
    cases = (
        (
            "用3,6,0三个数字可以组成()个不同的两位数.",
            ("2", "3", "4", "6"),
            2,
        ),
        (
            "用2,3,4,5四个数中的任意两个数求和,得数有()种可能.",
            ("6", "8", "5", "12"),
            2,
        ),
        (
            "小红、小丽、小刚三个同学排队,有()种排法.",
            ("4", "5", "6", "3"),
            2,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_capacity_inventory_time_and_geometry_templates_from_1918_run() -> None:
    cases = (
        (
            "图书角本来有36本书,借走15本后又还回来9本,还剩()本.",
            ("31", "30", "36", "21"),
            1,
        ),
        (
            "李阿姨做一件西服要用2米布,那么15米布最多可以做()件西服.",
            ("9", "6", "8", "7"),
            3,
        ),
        (
            "一本连环画3元,小熊拿着20元最多能买()本连环画?",
            ("6", "7", "4", "5"),
            0,
        ),
        (
            "15个小朋友坐船,每条船限乘4人,最少需要()条船.",
            ("4", "5", "3", "2"),
            0,
        ),
        (
            "现在是6:30,过一刻钟是().",
            ("06:15", "06:45", "7:00", "06:35"),
            1,
        ),
        ("一刻钟是().", ("60分", "10分", "30分", "15分"), 3),
        (
            "时针在3和4之间,分针指着6,此时的时间是().",
            ("04:30", "03:30", "4:60", "3:06"),
            1,
        ),
        (
            "5个正方形一共有()个直角.",
            ("20", "5", "25", "10"),
            0,
        ),
        (
            "长方形和正方形分别有()个直角.",
            ("4", "2", "8", "6"),
            0,
        ),
        (
            "角的大小和顶点的位置().",
            ("有关", "不确定", "无关", "顶点越高角越大"),
            2,
        ),
        (
            "一个三角形中最多可以有()个锐角,()个直角.",
            ("3,2", "2,1", "3,1", "2,2"),
            2,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_equal_group_wording_from_1918_run_is_local() -> None:
    cases = (
        (
            "吃饭时每人需要一双筷子,4个人需要()根筷子.",
            ("8", "4", "5", "6"),
            0,
        ),
        (
            "一只蜻蜓6条腿,4只蜻蜓()条腿.",
            ("25", "10", "24", "30"),
            2,
        ),
        (
            "每本笔记本5元,9本笔记本一共()元.",
            ("50", "45", "40", "35"),
            1,
        ),
        (
            "一只熊猫每天吃8根竹子,8天吃()根竹子.",
            ("81", "17", "64", "72"),
            2,
        ),
        (
            "姐姐带了一些糖果分给小朋友,5个小朋友正好每人分到了7个,"
            "姐姐一共带了()个糖果.",
            ("32", "25", "35", "12"),
            2,
        ),
        (
            "每个盒子可以装3个乒乓球,如果用4个这样的盒子来装10个乒乓球,"
            "结果().",
            ("不能确定", "装不下这些乒乓球", "没有全部装满", "正好装满"),
            2,
        ),
        (
            "把36个苹果摆在盘子里,每盘苹果的数量相同,每盘苹果的个数不可能是().",
            ("4个", "5个", "9个", "6个"),
            1,
        ),
        (
            "一块儿橡皮8角钱,买6块儿橡皮要用多少钱?用()计算比较简便.",
            ("减法", "加法", "除法", "乘法"),
            3,
        ),
        (
            "每只小熊6个玉米,8只小熊的玉米比50个().",
            ("一样多", "少", "不能确定", "多"),
            1,
        ),
        (
            "下列乘法口诀中,只能用来计算一个乘法算式的是().",
            ("三五十五", "五八四十", "六六三十六", "二三得六"),
            2,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_expression_choice_regressions_from_latest_run_are_local() -> None:
    cases = (
        (
            "下面算式中结果等于6的是().",
            ("52-15", "28-22", "89-80", "28-21"),
            1,
        ),
        (
            "下面算式的得数比60小的是().",
            ("90-15", "85-5", "55+10", "23+15"),
            3,
        ),
        (
            "得数大于50的算式是().",
            ("26+14=", "76-34=", "80-18=", "36+13"),
            2,
        ),
        (
            "5+5+5+4,不可以改写成算式().",
            ("5*4", "5*3+4", "4*5-1", "4+5*3"),
            0,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_extended_place_value_regressions_from_latest_run_are_local() -> None:
    cases = (
        ("下面各数中最接近7000的是().", ("6998", "7002", "7001", "6997"), 2),
        ("下面各数中最接近6000的是().", ("5999", "6002", "5997", "6004"), 0),
        (
            "下列四个数中,最接近20万的是:",
            ("199830", "200180", "199991", "200250"),
            2,
        ),
        (
            "下列四个数中,最接近7万的是:",
            ("69992", "70088", "70130", "69850"),
            0,
        ),
        ("从300起,一百一百地数,下一个数是().", ("400", "301", "310", "200"), 0),
        (
            "一百一百地数,数到3900,下一个数是().",
            ("3901", "4000", "3910", "3800"),
            1,
        ),
        (
            "用8、0、0、1可以组成的最大的四位数是().",
            ("8100", "8010", "8001", "1800"),
            0,
        ),
        ("“3”在十位上的数是().", ("634", "346", "364", "163"), 0),
        ("6345中的3表示().", ("3个百", "3个千", "3个十", "3个一"), 0),
        (
            "各数中的8表示8个百的是().",
            ("8606", "6806", "5608", "2086"),
            1,
        ),
        (
            "比6000小,比3500大的数是().",
            ("6500", "3050", "3500", "5098"),
            3,
        ),
        (
            "下面各数中只读一个“零”的是().",
            ("3600", "3000", "3006", "3660"),
            2,
        ),
        (
            "下面各数中一个“零”也不读的是().",
            ("2010", "3012", "3210", "3001"),
            2,
        ),
        (
            "下面各数中一个零都不读的是().",
            ("4005", "4500", "4055", "4050"),
            1,
        ),
        (
            "4080的两个“0”().",
            ("只读一个零", "都不读", "都读", "不能确定"),
            0,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_division_and_capacity_regressions_from_latest_run_are_local() -> None:
    cases = (
        (
            "除数是5,被除数是20,正确的算式是().",
            ("5/20=4", "5*4=20", "20/5=4", "20-5=15"),
            2,
        ),
        ("被除数是28,除数是7,商是().", ("35", "21", "5", "4"), 3),
        (
            "求8里面有几个2,列式是().",
            ("8/2=4", "8-2=6", "8+2=10", "8*2=16"),
            0,
        ),
        (
            "把20个苹果平均放进5个盘子,正确的算式是().",
            ("5*4=20", "20/5=4", "20-5=15", "20+5=25"),
            1,
        ),
        ("在□/△=○......5中,除数最小是().", ("5", "6", "4", "7"), 1),
        (
            "□/5=7......□中,余数可以是().",
            ("1,2,3,4", "0,1,2,3,4", "1,2,3,4,5", "0,1,2,3"),
            0,
        ),
        (
            "下面算式中余数是3的是().",
            ("48/7", "36/7", "48/5", "22/4"),
            2,
        ),
        (
            "计算32+3*6时,应先算().",
            ("32+3", "3*6", "32+6", "35*6"),
            1,
        ),
        (
            "要使47个气球能正好平均分给8个孩子,至少应该添加()个.",
            ("1", "2", "4", "3"),
            0,
        ),
        (
            "一条船能坐4人,33人至少需要()条船.",
            ("7", "8", "9", "10"),
            2,
        ),
        (
            "3个老师和16个同学去郊游,每辆出租车坐4人,至少需要()辆.",
            ("5", "4", "6", "7"),
            0,
        ),
        (
            "一根绳子27米,每4米做一根跳绳,最多能做()根跳绳.",
            ("6", "8", "5", "7"),
            0,
        ),
        (
            "26个学生去划船,每条船最多坐6人,至少需要()条船.",
            ("4", "6", "5", "7"),
            2,
        ),
        (
            "每支钢笔9元钱,张老师拿40元钱,最多能买()支.",
            ("3", "5", "6", "4"),
            3,
        ),
        (
            "50只兔子,每个笼子关8只,至少需要()个笼子.",
            ("6", "7", "8", "5"),
            1,
        ),
        (
            "每5人一桌,有27人,至少需要()张桌子.",
            ("7", "6", "5", "8"),
            1,
        ),
        (
            "从63中连续减去9,减()次才能使结果等于0.",
            ("6", "7", "8", "9"),
            1,
        ),
        (
            "从63里面连续减7,减()次后结果为零.",
            ("7", "8", "9", "10"),
            2,
        ),
        (
            "从40里面连续减去8,减()次后得数是0.",
            ("4", "6", "7", "5"),
            3,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_decimal_regressions_from_20260727_run_are_local() -> None:
    cases = (
        ("1.39中,百分位上的数是", ("3", "7", "1", "9"), 3),
        ("5.02中,百分位上的数是", ("5", "0", "2", "7"), 2),
        ("9.41中,十分位上的数是", ("9", "5", "4", "1"), 2),
        ("6.70中,百分位上的数是", ("0", "8", "7", "6"), 0),
        ("把3.45精确到十分之一是:", ("3.5", "4", "3", "3.4"), 0),
        ("把0.994精确到百分之一是:", ("1.00", "0.99", "1", "0.9"), 1),
        ("把2.995精确到百分之一是:", ("3", "2.99", "3.00", "2.9"), 2),
        ("把0.96精确到十分之一是:", ("1.0", "0.9", "0.90", "1"), 0),
        (
            "不改变数的大小,下面的数中可以去掉“0”的是:",
            ("4.05", "450", "4.50", "40.5"),
            2,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_advanced_arithmetic_regressions_from_20260727_run_are_local() -> None:
    cases = (
        (
            "下列各数中,与902*29的积最接近的是",
            ("27000", "32000", "36000", "24000"),
            0,
        ),
        (
            "下列各数中,与49*92的积最接近的是",
            ("4500", "4000", "5400", "6300"),
            0,
        ),
        (
            "下列各数中,与605*42的积最接近的是",
            ("20000", "30000", "28000", "24000"),
            3,
        ),
        (
            "下面算式中,去掉括号不改变计算结果的是",
            ("52/(4*13)", "(300/6)-20", "87-(47-27)", "(65-35)*4"),
            1,
        ),
        (
            "下面算式中,去掉括号不改变计算结果的是",
            ("63/(7*3)", "84-(42-21)", "(210/7)-20", "(54-27)*5"),
            2,
        ),
        (
            "下面算式中,去掉括号不改变运算顺序的算式是哪一个?",
            (
                "(18*69)-(50/25)",
                "18*[(69-50)+25]",
                "1000-480/(4*8)",
                "(18+69)*(50-25)",
            ),
            0,
        ),
        ("若A*6=4.8,则A*60=", ("0.48", "48", "480", "4.8"), 1),
        ("若A*20=180,则A*200=", ("1800", "180", "18", "1.8"), 0),
        (
            "用竖式计算387*65时,乘数65十位上的6乘387得",
            ("23220", "3870", "387", "2322"),
            0,
        ),
        (
            "用竖式计算512*47时,乘数47十位上的4乘512得",
            ("512", "2048", "20480", "5120"),
            2,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_large_place_value_composition_from_20260727_run_is_local() -> None:
    cases = (
        (
            "三千零四十万写作:",
            ("30040000", "30400000", "300040", "3000040"),
            1,
        ),
        (
            "3个千万、7个百、9个一组成的数是",
            ("30000709", "30000790", "30070009", "30007009"),
            0,
        ),
        (
            "6个十万、3个百、7个十组成的数是:",
            ("603070", "637000", "6003007", "600370"),
            3,
        ),
        (
            "5个千万、6个百万、8个十、3个一组成的数是",
            ("56008003", "56000083", "56000803", "50600083"),
            1,
        ),
        (
            "9个亿、4个千、8个一组成的数是",
            ("90004008", "900000408", "900004008", "900004080"),
            2,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text


def test_minute_hand_rotation_from_20260727_run_is_local() -> None:
    cases = (
        (
            "从8:00走到8:40,分针转动了",
            ("240°", "180°", "120°", "90°"),
            0,
        ),
        (
            "从12:00走到12:28,分针转动了",
            ("180°", "84°", "168°", "120°"),
            2,
        ),
        ("从3:00走到3:15,分针转动了_度.", ("60", "90", "120", "15"), 1),
        (
            "从9:00走到9:45,分针转动了",
            ("90°", "135°", "270°", "180°"),
            2,
        ),
    )
    engine = RuleEngine()
    for text, options, expected_index in cases:
        result = engine.solve(question(text, options))
        assert result is not None, text
        assert result.answer_index == expected_index, text
