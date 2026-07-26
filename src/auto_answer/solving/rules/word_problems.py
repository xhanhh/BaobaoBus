"""Deterministic arithmetic word-problem rules."""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique

_NUMBER_TOKEN = re.compile(NUMBER)
_MAKE_MAXIMUM_GROUPS = re.compile(
    rf"(?:做|制作)(?:一个|1个)?[^,，。]*?(?:要用|需要)"
    rf"(?P<per_group>{NUMBER})[^,，。]*?[,，]"
    rf"(?P<total>{NUMBER})[^,，。]*?最多(?:可以)?(?:做|制作)"
)
_REPEATED_ACTION = re.compile(
    rf"每次(?:拿|取|搬|吃|用)(?:掉|去)?(?P<per_action>{NUMBER})"
    rf"[^,，。]*?[,，][^,，。]*?(?:拿|取|搬|吃|用)(?:了)?"
    rf"(?P<times>{NUMBER})次[^,，。]*?[,，][^。]*?一共"
)
_COMPLETE_GROUPS = re.compile(
    rf"(?:有|共|一共有)(?P<total>{NUMBER})[^,，。]*?[,，]"
    rf"每(?P<per_group>{NUMBER})[^,，。]*?(?:装|放|分|扎)[^,，。]*?"
    rf"[,，](?:可以|能)?(?:装满|装|分成|扎成|扎)"
)
_CAPACITY_REQUIRED = re.compile(
    rf"(?:有|共|一共有)(?P<total>{NUMBER})[^,，。]*?[,，]"
    rf"每(?:条船|辆车|个盒子?|个袋子?|盒|袋)"
    rf"(?:坐|装|放|容纳)(?P<per_group>{NUMBER})"
    rf"[^,，。]*?[,，][^。]*?需要"
)
_CAPACITY_ITEMS_FIRST = re.compile(
    rf"(?:有|共|一共有)(?P<total>{NUMBER})[^,，。]*?[,，]"
    rf"每(?P<per_group>{NUMBER})[^,，。]*?装一个(?:笼子|盒子|袋子)"
    rf"[^,，。]*?[,，][^。]*?需要"
)
_INITIAL_INVENTORY = re.compile(
    rf"(?:原来有|本来有|原有|有|买回来|买来|买了|摘了|采了|"
    rf"(?:公交车|汽车|车)上有(?:乘客)?)(?P<initial>{NUMBER})"
    r"(?:个|根|块|颗|本|张|只|米|人)"
)
_INVENTORY_ACTION = re.compile(
    rf"(?P<verb>吃掉了?|吃了|用了|用去了?|拿走了?|送了|放飞了?|飞走了?|"
    rf"破了?|爆了?|坏了?|丢了?|损坏了?|借走了?|"
    rf"卖了|走了|下了|下车了?|搬走了?|又买了|买来了?|又摘了?|又来了?|"
    rf"又搬来了?|又还回来|还回来了?|"
    rf"增加了?|添了|上了|上车了?)(?P<amount>{NUMBER})"
    r"(?:个|根|块|颗|本|张|只|米)?"
)
_NEGATIVE_ACTION = re.compile(
    rf"(?:吃掉了?|吃了|用了|用去了?|拿走了?|送了|放飞了?|飞走了?|"
    rf"破了?|爆了?|坏了?|丢了?|损坏了?|借走了?|"
    rf"卖了|走了|下了|下车了?|搬走了?)"
    rf"(?P<amount>{NUMBER})(?:个|根|块|颗|本|张|只|米)?"
)
_REMAINING_QUERY = re.compile(
    r"还剩|现在(?:还)?(?:有|剩)|现在车上有|目前(?:还)?(?:有|剩)"
)
_POSITIVE_ACTIONS = (
    "又买了",
    "买来",
    "又摘",
    "又来",
    "又搬来",
    "又还回来",
    "还回来",
    "增加",
    "添了",
    "上了",
    "上车",
)
_TWO_RECIPIENT_DISTRIBUTION = re.compile(
    rf"(?:有|摘了)?(?P<total>{NUMBER})(?:个)?.*?"
    rf"分给两(?:只|个).*?(?:几|多少)种分法.*?"
    rf"每(?:只|个)[^,，。]*?最少分1个"
)
_GROUP_PHOTO_COUNT = re.compile(
    rf"和小组里的每一个同学都合照一次[^,，。]*[,，]"
    rf"一共照了(?P<count>{NUMBER})次.*?小组里一共有多少人"
)
_TWO_PERSON_COMBINED_TOTAL = re.compile(
    rf"(?P<first_name>[\u4e00-\u9fff]{{1,4}})有(?P<base>{NUMBER})[^,，。]*[,，]"
    rf"[\u4e00-\u9fff]{{1,4}}(?:的[^,，。]*)?比(?P=first_name)"
    rf"(?P<direction>少|多)(?P<delta>{NUMBER})[^,，。]*[,，]"
    rf"两人一共有"
)
_CORRECT_REMAINING_EQUATION = re.compile(
    rf"(?:买回来|原来有|有)(?P<initial>{NUMBER})[^,，。]*[,，]"
    rf"[^,，。]*?(?:用(?:了|去)|吃了?|卖了?|拿走了?)(?P<used>{NUMBER})"
    rf"[^,，。]*[,，][^。]*?还剩.*?正确的算式"
)
_PEOPLE_BEHIND = re.compile(
    rf"(?P<total>{NUMBER})个[^,，。]*?(?:排队|排成一排)[^,，。]*[,，]"
    rf"[^,，。]*?前面有(?P<front>{NUMBER})(?:个)?人[^,，。]*[,，]"
    rf"[^,，。]*?后面有(?:\(\)|多少|几)(?:个)?人?"
)
_EAGLE_CATCHING_CHICKS = re.compile(
    rf"(?P<total>{NUMBER})个小朋友玩老鹰捉小鸡.*?"
    rf"捉到了(?P<caught>{NUMBER})只小鸡.*?还有(?:\(\)|多少|几)只小鸡没"
)
_POWER_OUTAGE_LIGHTS = re.compile(
    r"(?:灯|电灯).*?(?:突然)?停电了.*?还有(?:\(\)|多少|几).*?灯亮着"
)
_EQUAL_TRANSFER = re.compile(
    rf"(?P<first_name>[\u4e00-\u9fff]{{1,4}})(?:有|做了|画了)(?P<first>{NUMBER})"
    rf"[^,，。]*[,，](?P<second_name>[\u4e00-\u9fff]{{1,4}})(?:有|做了|画了)"
    rf"(?P<second>{NUMBER})[^,，。]*[,，]"
    rf"(?P<giver>[\u4e00-\u9fff]{{1,4}})给(?P<receiver>[\u4e00-\u9fff]{{1,4}})"
    rf"(?:\(\)|多少|几)[^,，。]*[,，].*?(?:一样多|相等)"
)
_TRANSFER_WITH_REMAINING_DIFFERENCE = re.compile(
    rf"(?P<first_name>[\u4e00-\u9fff]{{1,4}})有(?P<first>{NUMBER})"
    rf"(?P<unit>个|本|张|支|颗|元)[^,，。]*[,，]"
    rf"(?P<second_name>[\u4e00-\u9fff]{{1,4}})有(?P<second>{NUMBER})"
    rf"(?P=unit)[^,，。]*[,，]"
    rf"(?P<giver>[\u4e00-\u9fff]{{1,4}})给"
    rf"(?P<receiver>[\u4e00-\u9fff]{{1,4}})(?:\(\)|多少|几)(?P=unit)"
    rf"[,，](?P=giver)还比(?P=receiver)"
    rf"(?P<direction>多|少)(?P<remaining>{NUMBER})(?P=unit)"
)
_REVERSE_DIFFERENCE = re.compile(
    rf"(?P<larger>{NUMBER})比(?:\(\)|多少|几)(?P<direction>多|少)"
    rf"(?P<delta>{NUMBER})"
)
_TWO_EQUAL_ADDENDS = re.compile(
    rf"两个加数都是(?P<addend>{NUMBER})[^。]*?和是(?:\(\)|多少|几)"
)
_BOX_TOTAL = re.compile(
    rf"(?:有|买了)(?P<count>{NUMBER})(?:盒|袋)[^,，。]*[,，]"
    rf"每(?:盒|袋)(?P<per>{NUMBER})[^,，。]*[,，].*?一共(?:有|买了)"
)
_TWO_AND_HALF_BOXES = re.compile(
    rf"一盒[^,，。]*?(?P<per>{NUMBER})[^,，。]*[,，]"
    rf"两盒半[^,，。]*?一共有"
)
_TWO_CATEGORY_TOTAL = re.compile(
    rf"(?:男生|男同学)(?P<first>{NUMBER})人[^,，。]*[,，]"
    rf"(?:女生|女同学)(?P<second>{NUMBER})人"
    rf"[^,，。]*[,，].*?一共有(?:\(\)|多少|几)人"
)
_OLDER_SIBLING_FUTURE_AGE = re.compile(
    rf"(?:我|弟弟|妹妹)今年(?P<younger>{NUMBER})岁[,，]"
    rf"(?:姐姐|哥哥)比(?:我|弟弟|妹妹)大(?P<older>{NUMBER})岁[,，]"
    rf"(?P<years>{NUMBER})年后(?:姐姐|哥哥)(?:\(\)|多少|几)岁"
)
_REPEATED_SUBTRACTION_TO_ZERO = re.compile(
    rf"从(?P<total>{NUMBER})里连续减去(?P<amount>{NUMBER})[,，]"
    rf"减(?:\(\)|多少|几)次后结果是0"
)
_SAME_AMOUNT_TWO_PERIODS = re.compile(
    rf"(?:上午|昨天)[^,，。]*?(?P<amount>{NUMBER})[^,，。]*[,，]"
    rf"(?:下午|今天)[^,，。]*?(?:和|与)(?:上午|昨天)(?:同样多|一样多)"
    rf"[^,，。]*[,，].*?(?:一天|两天)(?:一共)?[^。]*?(?:\(\)|多少|几)"
)
_RIGHT_SIDE_COUNT = re.compile(
    rf"(?P<total>{NUMBER})个同学排队[^,，。]*[,，]"
    rf"[^,，。]*?左边有(?P<left>{NUMBER})人[^,，。]*[,，]"
    rf"(?:他|她)?右边有(?:\(\)|多少|几)人"
)
_AGE_DIFFERENCE = re.compile(
    rf"(?P<younger_name>[\u4e00-\u9fff]{{1,4}})(?:今年)?"
    rf"(?P<younger>{NUMBER})岁[,，]"
    rf"(?P<older_name>[\u4e00-\u9fff]{{1,4}})(?P<older>{NUMBER})岁[,，]"
    rf"(?P=older_name)比(?P=younger_name)大(?:\(\)|多少|几)岁"
)
_WEIGHT_INVENTORY = re.compile(
    rf"重(?P<initial>{NUMBER})千克[,，]"
    rf"吃了(?P<used>{NUMBER})千克[,，]"
    rf"又买来(?P<added>{NUMBER})千克[,，]"
    rf"现在重(?:\(\)|多少|几)千克"
)
_RESOURCE_MAXIMUM = re.compile(
    rf"做一件[^,，。]*?要用(?P<per>{NUMBER})(?P<unit>米|个|根|张)"
    rf"[^,，。]*[,，][^0-9]*?(?P<total>{NUMBER})(?P=unit)"
    rf"[^。]*?最多可以做(?:\(\)|多少|几)件"
)
_MINIMUM_TO_EXCEED = re.compile(
    rf"(?P<leader_name>[\u4e00-\u9fff]{{1,4}})(?:做了|有)"
    rf"(?P<leader>{NUMBER})[^,，。]*[,，]"
    rf"(?P<trailer_name>[\u4e00-\u9fff]{{1,4}})(?:做了|有)"
    rf"(?P<trailer>{NUMBER})[^。]*?"
    rf"(?P=trailer_name)至少还要(?:做|得|拿|增加)"
    rf"(?:\(\)|多少|几)[^。]*?才能超过(?P=leader_name)"
)
_TWO_CONTAINER_REMAINDER = re.compile(
    rf"有(?P<total>{NUMBER})个[^,，。]*[,，]"
    rf"放在两个(?P<container>篮子|盒子|袋子)里[,，]"
    rf"一个(?P=container)放(?P<first>{NUMBER})个[,，]"
    rf"另一个(?P=container)放(?:\(\)|多少|几)个"
)
_HALF_CONTENT_CONTAINER_WEIGHT = re.compile(
    rf"连(?P<container>桶|筐|盒)重(?P<full>{NUMBER})千克[,，]"
    rf"用去一半[^,，。]*后[,，]"
    rf"连(?P=container)重(?P<half>{NUMBER})千克[,，]"
    rf"(?P=container)重(?:\(\)|多少|几)千克"
)


def solve_counting_choice(question: Question) -> SolveDecision | None:
    supplies = re.search(
        rf"有(?P<students>{NUMBER})个学生和(?P<teachers>{NUMBER})个老师"
        rf".*?每人一瓶水[,，]准备(?P<prepared>{NUMBER})瓶水[,，]"
        rf"(?:够分配么|够不够)",
        question.text,
    )
    if supplies:
        required = Decimal(supplies.group("students")) + Decimal(
            supplies.group("teachers")
        )
        prepared = Decimal(supplies.group("prepared"))
        expected = "正好" if prepared == required else "够" if prepared > required else "不够"
        matches = [
            index for index, option in enumerate(question.options) if option == expected
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"supplies comparison: {prepared} vs {required}",
            )
        return None

    distribution = _TWO_RECIPIENT_DISTRIBUTION.search(question.text)
    if distribution:
        total = Decimal(distribution.group("total"))
        if total != total.to_integral_value() or total < 2:
            return None
        return _match_count_option(
            total - 1,
            question.options,
            suffix="种",
            reason="two distinct recipients",
        )

    photos = _GROUP_PHOTO_COUNT.search(question.text)
    if photos:
        count = Decimal(photos.group("count"))
        if count != count.to_integral_value() or count < 0:
            return None
        return _match_count_option(
            count + 1,
            question.options,
            suffix="人",
            reason="photographer plus photographed classmates",
        )

    remaining_equation = _CORRECT_REMAINING_EQUATION.search(question.text)
    if remaining_equation:
        initial = Decimal(remaining_equation.group("initial"))
        used = Decimal(remaining_equation.group("used"))
        remaining = initial - used
        if initial < 0 or used < 0 or remaining < 0:
            return None
        expected = f"{initial}-{used}={remaining}"
        matches = [
            index
            for index, option in enumerate(question.options)
            if option.replace(" ", "").strip(".。") == expected
        ]
        if len(matches) != 1:
            return None
        return SolveDecision(
            matches[0],
            "rule",
            f"remaining quantity equation: {expected}",
        )
    return None


def solve_word_problem(text: str) -> Decimal | None:
    resource_maximum = _RESOURCE_MAXIMUM.search(text)
    if resource_maximum:
        per = Decimal(resource_maximum.group("per"))
        total = Decimal(resource_maximum.group("total"))
        return total // per if total >= 0 and per > 0 else None

    age_difference = _AGE_DIFFERENCE.search(text)
    if age_difference:
        younger = Decimal(age_difference.group("younger"))
        older = Decimal(age_difference.group("older"))
        difference = older - younger
        return difference if younger >= 0 and difference >= 0 else None

    weight_inventory = _WEIGHT_INVENTORY.search(text)
    if weight_inventory:
        initial = Decimal(weight_inventory.group("initial"))
        used = Decimal(weight_inventory.group("used"))
        added = Decimal(weight_inventory.group("added"))
        target = initial - used + added
        return target if initial >= used >= 0 and added >= 0 else None

    minimum_to_exceed = _MINIMUM_TO_EXCEED.search(text)
    if minimum_to_exceed:
        leader = Decimal(minimum_to_exceed.group("leader"))
        trailer = Decimal(minimum_to_exceed.group("trailer"))
        needed = leader - trailer + 1
        return needed if leader >= trailer >= 0 else None

    two_containers = _TWO_CONTAINER_REMAINDER.search(text)
    if two_containers:
        total = Decimal(two_containers.group("total"))
        first = Decimal(two_containers.group("first"))
        remaining = total - first
        return remaining if total >= first >= 0 else None

    half_content = _HALF_CONTENT_CONTAINER_WEIGHT.search(text)
    if half_content:
        full = Decimal(half_content.group("full"))
        half = Decimal(half_content.group("half"))
        container = half * 2 - full
        return container if full >= half >= 0 and container >= 0 else None

    repeated_subtraction = _REPEATED_SUBTRACTION_TO_ZERO.search(text)
    if repeated_subtraction:
        total = Decimal(repeated_subtraction.group("total"))
        amount = Decimal(repeated_subtraction.group("amount"))
        if total < 0 or amount <= 0:
            return None
        count, remainder = divmod(total, amount)
        return count if remainder == 0 else None

    same_periods = _SAME_AMOUNT_TWO_PERIODS.search(text)
    if same_periods:
        amount = Decimal(same_periods.group("amount"))
        return amount * 2 if amount >= 0 else None

    future_age = _OLDER_SIBLING_FUTURE_AGE.search(text)
    if future_age:
        younger = Decimal(future_age.group("younger"))
        older = Decimal(future_age.group("older"))
        years = Decimal(future_age.group("years"))
        if younger < 0 or older < 0 or years < 0:
            return None
        return younger + older + years

    right_side = _RIGHT_SIDE_COUNT.search(text)
    if right_side:
        total = Decimal(right_side.group("total"))
        left = Decimal(right_side.group("left"))
        right = total - left - 1
        return right if total > 0 and left >= 0 and right >= 0 else None

    equal_addends = _TWO_EQUAL_ADDENDS.search(text)
    if equal_addends:
        return Decimal(equal_addends.group("addend")) * 2

    box_total = _BOX_TOTAL.search(text)
    if box_total:
        count = Decimal(box_total.group("count"))
        per = Decimal(box_total.group("per"))
        return count * per if count >= 0 and per >= 0 else None

    half_boxes = _TWO_AND_HALF_BOXES.search(text)
    if half_boxes:
        per = Decimal(half_boxes.group("per"))
        return per * Decimal("2.5") if per >= 0 else None

    category_total = _TWO_CATEGORY_TOTAL.search(text)
    if category_total:
        first = Decimal(category_total.group("first"))
        second = Decimal(category_total.group("second"))
        return first + second if first >= 0 and second >= 0 else None

    if _POWER_OUTAGE_LIGHTS.search(text):
        return Decimal()

    remaining_difference = _TRANSFER_WITH_REMAINING_DIFFERENCE.search(text)
    if remaining_difference:
        first = Decimal(remaining_difference.group("first"))
        second = Decimal(remaining_difference.group("second"))
        giver = remaining_difference.group("giver")
        receiver = remaining_difference.group("receiver")
        if (
            giver == remaining_difference.group("first_name")
            and receiver == remaining_difference.group("second_name")
        ):
            initial_difference = first - second
        elif (
            giver == remaining_difference.group("second_name")
            and receiver == remaining_difference.group("first_name")
        ):
            initial_difference = second - first
        else:
            return None
        desired_difference = Decimal(remaining_difference.group("remaining"))
        if remaining_difference.group("direction") == "少":
            desired_difference = -desired_difference
        transfer = (initial_difference - desired_difference) / 2
        if (
            first < 0
            or second < 0
            or transfer < 0
            or transfer != transfer.to_integral_value()
        ):
            return None
        return transfer

    eagle_game = _EAGLE_CATCHING_CHICKS.search(text)
    if eagle_game:
        total = Decimal(eagle_game.group("total"))
        caught = Decimal(eagle_game.group("caught"))
        remaining = total - 2 - caught
        return remaining if total >= 2 and caught >= 0 and remaining >= 0 else None

    equal_transfer = _EQUAL_TRANSFER.search(text)
    if equal_transfer:
        first = Decimal(equal_transfer.group("first"))
        second = Decimal(equal_transfer.group("second"))
        first_name = equal_transfer.group("first_name")
        second_name = equal_transfer.group("second_name")
        giver = equal_transfer.group("giver")
        receiver = equal_transfer.group("receiver")
        if giver == first_name and receiver == second_name:
            transfer = (first - second) / 2
        elif giver == second_name and receiver == first_name:
            transfer = (second - first) / 2
        else:
            return None
        return transfer if first >= 0 and second >= 0 and transfer >= 0 else None

    reverse_difference = _REVERSE_DIFFERENCE.search(text)
    if reverse_difference:
        larger = Decimal(reverse_difference.group("larger"))
        delta = Decimal(reverse_difference.group("delta"))
        if delta < 0:
            return None
        return (
            larger - delta
            if reverse_difference.group("direction") == "多"
            else larger + delta
        )

    combined_total = _TWO_PERSON_COMBINED_TOTAL.search(text)
    if combined_total:
        base = Decimal(combined_total.group("base"))
        delta = Decimal(combined_total.group("delta"))
        if base < 0 or delta < 0:
            return None
        second = (
            base - delta
            if combined_total.group("direction") == "少"
            else base + delta
        )
        if second < 0:
            return None
        return base + second

    people_behind = _PEOPLE_BEHIND.search(text)
    if people_behind:
        total = Decimal(people_behind.group("total"))
        front = Decimal(people_behind.group("front"))
        behind = total - front - 1
        return behind if total > 0 and front >= 0 and behind >= 0 else None

    repeated_action = _REPEATED_ACTION.search(text)
    if repeated_action:
        per_action = Decimal(repeated_action.group("per_action"))
        times = Decimal(repeated_action.group("times"))
        if per_action < 0 or times < 0:
            return None
        return per_action * times

    complete_groups = _COMPLETE_GROUPS.search(text)
    if complete_groups:
        total = Decimal(complete_groups.group("total"))
        per_group = Decimal(complete_groups.group("per_group"))
        if total < 0 or per_group <= 0:
            return None
        return total // per_group

    capacity_required = _CAPACITY_REQUIRED.search(text)
    if capacity_required:
        total = Decimal(capacity_required.group("total"))
        per_group = Decimal(capacity_required.group("per_group"))
        if total < 0 or per_group <= 0:
            return None
        full_groups, remainder = divmod(total, per_group)
        return full_groups + (1 if remainder else 0)

    capacity_items_first = _CAPACITY_ITEMS_FIRST.search(text)
    if capacity_items_first:
        total = Decimal(capacity_items_first.group("total"))
        per_group = Decimal(capacity_items_first.group("per_group"))
        if total < 0 or per_group <= 0:
            return None
        full_groups, remainder = divmod(total, per_group)
        return full_groups + (1 if remainder else 0)

    maximum_groups = _MAKE_MAXIMUM_GROUPS.search(text)
    if maximum_groups:
        per_group = Decimal(maximum_groups.group("per_group"))
        total = Decimal(maximum_groups.group("total"))
        if per_group <= 0 or total < 0:
            return None
        return total // per_group

    inventory = _inventory_total(text)
    if inventory is not None:
        return inventory

    if (
        "短了" in text
        or "一共用去" in text
        or "一共吃了" in text
        or "少了" in text
    ):
        reductions = [
            Decimal(match.group("amount"))
            for match in _NEGATIVE_ACTION.finditer(text)
        ]
        if reductions:
            return sum(reductions, Decimal())

    currency_exchange = re.search(
        rf"(?:(\d+)张)?({NUMBER})元可以换(?:\(\)|多少|几)?张({NUMBER})元",
        text,
    )
    if currency_exchange:
        note_count = Decimal(currency_exchange.group(1) or "1")
        source_value = Decimal(currency_exchange.group(2))
        target_value = Decimal(currency_exchange.group(3))
        if target_value == 0:
            return None
        return note_count * source_value / target_value

    queue = re.search(
        rf"(?:有|一共有)?({NUMBER})个.*?(?:排队|排成一排).*?"
        rf"从左数.*?第({NUMBER})(?:个)?.*?从右数.*?第(?:\(\)|多少|几)",
        text,
    )
    if queue:
        return Decimal(queue.group(1)) - Decimal(queue.group(2)) + 1

    younger = re.search(
        rf"({NUMBER})岁.*?比(?:爸爸|妈妈).*?小({NUMBER})岁.*?"
        rf"(?:爸爸|妈妈).*?\(\)岁",
        text,
    )
    if younger:
        return Decimal(younger.group(1)) + Decimal(younger.group(2))

    patterns: tuple[tuple[str, Callable[[Decimal, Decimal], Decimal]], ...] = (
        (rf"比({NUMBER})多({NUMBER})(?:的数)?", operator.add),
        (rf"比({NUMBER})少({NUMBER})(?:的数)?", operator.sub),
    )
    for pattern, operation in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return operation(Decimal(match.group(1)), Decimal(match.group(2)))
            except (InvalidOperation, ArithmeticError):
                return None
    return None


def _inventory_total(text: str) -> Decimal | None:
    remaining_query = _REMAINING_QUERY.search(text)
    if remaining_query is None:
        return None
    initial_match = _INITIAL_INVENTORY.search(text)
    if initial_match is None:
        return None

    total = Decimal(initial_match.group("initial"))
    actions = list(_INVENTORY_ACTION.finditer(text, initial_match.end()))
    if not actions:
        return None
    # Never silently skip an unknown numeric action. A partial calculation is
    # more dangerous than declining the rule and handing the question to LLM.
    amount_spans = [
        (action.start("amount"), action.end("amount")) for action in actions
    ]
    for number in _NUMBER_TOKEN.finditer(
        text,
        initial_match.end(),
        remaining_query.start(),
    ):
        if not any(
            start <= number.start() and number.end() <= end
            for start, end in amount_spans
        ):
            return None
    for action in actions:
        amount = Decimal(action.group("amount"))
        verb = action.group("verb")
        if any(verb.startswith(token) for token in _POSITIVE_ACTIONS):
            total += amount
        else:
            total -= amount
    return total


def _match_count_option(
    target: Decimal,
    options: tuple[str, str, str, str],
    *,
    suffix: str,
    reason: str,
) -> SolveDecision | None:
    values: list[Decimal | None] = []
    for option in options:
        match = re.fullmatch(rf"({NUMBER}){re.escape(suffix)}", option.strip())
        values.append(Decimal(match.group(1)) if match else None)
    return match_unique(target, tuple(values), reason)
