"""Deterministic Chinese display translations for the blind-review prompts.

The source prompt remains unchanged in the review bundle.  This module only
creates a reviewer-facing Chinese rendering for the known VBench-derived cases;
it does not infer scores, alter the prompt used by the harness, or expose arm
metadata.
"""

from __future__ import annotations

import re


SCENE_ZH = {
    "vbench2-camera_motion-017": "笔记本电脑，镜头推进。",
    "vbench2-complex_plot-079": (
        "在一个被遗忘的古老村庄里，所有钟表都停了，时间仿佛静止。村民被困在永恒的闲散状态中。"
        "年轻的钟表匠 Daniel 决定找出原因并修复钟楼。他发现钟楼的核心被一只巨大的机械鸟偷走，"
        "机械鸟藏在地下宫殿。Daniel 深入宫殿，与怪物战斗并躲避陷阱，最终找到并修复机械鸟。"
        "修好机械鸟的机关后，他恢复了村庄的时间流动。然而 Daniel 意识到，每次恢复时间，村庄的一段记忆就会消失。"
        "最后，他选择牺牲自己的记忆来重新启动时间，为村民带来新的开始。"
    ),
    "vbench2-dynamic_spatial_relationship-095": "一只鸟在树的前方，然后飞到树的上方。",
    "vbench2-human_interaction-035": "一个人把饮料递给另一个人。",
    "vbench2-motion_order-060": "一只狗正在追球，然后突然停下并躺下休息。",
    "vbench2-camera_motion-005": "富士山，镜头向右摇摄。",
    "vbench2-complex_plot-075": (
        "在一座神秘的古墓中，据说藏着一件能够改变历史的宝物。年轻考古学家 Zhang Peng 与一群盗墓者进入古墓。"
        "他们发现墙上刻着奇怪符号，似乎在警告禁入区域。Zhang Peng 凭知识解读符号，但接踵而来的陷阱将他们置于危险中。"
        "经历多次失败后，Zhang Peng 和盗墓者终于找到神秘遗物。然而取走遗物时，一阵强风把他们送入未知空间。"
        "最后 Zhang Peng 意识到，这件物品不是财富，而是一枚封印，里面封存着古老诅咒，揭开了更大的阴谋。"
    ),
    "vbench2-dynamic_spatial_relationship-097": "一只猴子在苹果的右侧，然后跳到苹果的左侧。",
    "vbench2-human_interaction-027": "一个人把一块蛋糕递给另一个人。",
    "vbench2-motion_order-055": "一个人正在吃晚餐，然后突然开始洗餐具。",
    "vbench2-camera_motion-008": "桌子，镜头向下俯拍。",
    "vbench2-complex_plot-070": (
        "丑小鸭出生在温暖的鸭子家庭，却因外貌不同而遭到其他小鸭排斥。它决定离家出走，踏上寻找归属的旅程。"
        "一路上它经历许多困难，感到孤独悲伤。熬过寒冬后，春天里它发现自己变成了美丽的天鹅。"
        "丑小鸭终于找到真正的家，与其他天鹅一同翱翔，成为最引人注目的那只。它明白，人不应被他人的看法定义，而应相信内在的美。"
    ),
    "vbench2-dynamic_spatial_relationship-096": "一只猴子在苹果的后方，然后跳到苹果的前方。",
    "vbench2-human_interaction-034": "两个人正在把气球系在一起。",
    "vbench2-motion_order-053": "一匹马沿着小路行走，然后突然开始奔跑。",
    "vbench2-camera_motion-014": "马丘比丘，镜头拉远。",
    "vbench2-complex_plot-067": (
        "一个小男孩收到来自未来的信，信中说他将发现一个改变世界的秘密。第一天，他找到一本写满未知符号的尘封书。"
        "第二天，他在书中找到一张地图，指引他前往一座废弃城堡。第三天，他进入城堡，发现一间封闭房间，里面有一座古老时钟。"
        "第四天，他遵循时钟指示启动机关，时钟的秒针开始倒转。第五天，他意识到世界的秘密不在外部世界，而在于他如何看待过去和未来。"
        "这个秘密让他能够活在当下，不再受时间束缚。"
    ),
    "vbench2-dynamic_spatial_relationship-089": "一只狐狸在鞋子的右侧，然后跑到鞋子的左侧。",
    "vbench2-human_interaction-038": "一个人帮助另一个人摆放餐桌。",
    "vbench2-motion_order-048": "一个人正在喝茶，然后突然开始叠衣服。",
    "vbench2-camera_motion-001": "花园，镜头推进。",
    "vbench2-complex_plot-080": (
        "遥远的山脉中据说有一条传说中的红龙。每当它孵化一枚金蛋，世界的财富与幸运就会爆发。"
        "一位名叫 Leon 的年轻冒险家听说传说后决定寻找红龙。一路上他遭遇许多危险，但凭借决心最终找到龙巢并成功偷走金蛋。"
        "然而 Leon 带着金蛋回到王国时，发现王国因他的行为陷入混乱。最后 Leon 决定归还金蛋，并用智慧恢复王国的和平。"
    ),
    "vbench2-dynamic_spatial_relationship-082": "一只狗在岩石的右侧，然后跑到岩石的左侧。",
    "vbench2-human_interaction-022": "两个人交换一本书。",
    "vbench2-motion_order-042": "一匹马正在田野里小跑，然后突然开始在溪流中喝水。",
    "vbench2-camera_motion-009": "阿尔罕布拉宫，镜头推进。",
    "vbench2-complex_plot-072": (
        "一座被火山环绕的岛屿上流传着火山神的传说。传说火山中蕴含不朽力量，每一代岛屿守护者都必须接受试炼才能获得神的力量。"
        "一位年轻守护者决定接受挑战并攀登火山。在火山深处，他面临生死抉择。最终，他用勇气和智慧击败火山内部的神秘敌人，"
        "获得神的力量，成功保护岛屿。"
    ),
    "vbench2-dynamic_spatial_relationship-094": "一只鸟在树的上方，然后飞到树的后方。",
    "vbench2-human_interaction-040": "一个人把相机递给另一个人拍照。",
    "vbench2-motion_order-057": "一只猫正在自我梳理，然后突然开始追逐一只虫子。",
}


DIMENSION_ZH = {
    "Camera_Motion": "摄像机运动",
    "Complex_Plot": "复杂情节",
    "Dynamic_Spatial_Relationship": "动态空间关系",
    "Human_Interaction": "人物交互",
    "Motion_Order_Understanding": "动作顺序理解",
}


CUE_ZH = {
    "vbench2-camera_motion-017": "镜头推进",
    "vbench2-camera_motion-005": "向右摇摄",
    "vbench2-camera_motion-008": "向下俯拍",
    "vbench2-camera_motion-014": "镜头拉远",
    "vbench2-camera_motion-001": "镜头推进",
    "vbench2-camera_motion-009": "镜头推进",
    "vbench2-dynamic_spatial_relationship-095": "鸟是否在树的前方（是或否），鸟是否在树的上方（是或否）",
    "vbench2-dynamic_spatial_relationship-097": "猴子是否在苹果右侧（是或否），猴子是否在苹果左侧（是或否）",
    "vbench2-dynamic_spatial_relationship-096": "猴子是否在苹果后方（是或否），猴子是否在苹果前方（是或否）",
    "vbench2-dynamic_spatial_relationship-089": "狐狸是否在鞋子右侧（是或否），狐狸是否在鞋子左侧（是或否）",
    "vbench2-dynamic_spatial_relationship-082": "狗是否在岩石右侧（是或否），狗是否在岩石左侧（是或否）",
    "vbench2-dynamic_spatial_relationship-094": "鸟是否在树的上方（是或否），鸟是否在树的后方（是或否）",
    "vbench2-human_interaction-035": "未指定",
    "vbench2-human_interaction-027": "未指定",
    "vbench2-human_interaction-034": "未指定",
    "vbench2-human_interaction-038": "未指定",
    "vbench2-human_interaction-022": "未指定",
    "vbench2-human_interaction-040": "未指定",
    "vbench2-motion_order-060": "追逐球、躺下休息",
    "vbench2-motion_order-055": "吃晚餐、清洗餐具",
    "vbench2-motion_order-053": "沿小路行走、奔跑",
    "vbench2-motion_order-048": "喝茶、叠衣服",
    "vbench2-motion_order-042": "在田野小跑、从溪流喝水",
    "vbench2-motion_order-057": "自我梳理、追逐虫子",
    "vbench2-complex_plot-079": "修复停摆的钟楼、寻找被机械鸟偷走的钟心、恢复村庄时间并牺牲记忆重启时间",
    "vbench2-complex_plot-075": "探索古墓、破解符号、躲避陷阱、发现封印并揭开阴谋",
    "vbench2-complex_plot-070": "丑小鸭被排斥、经历冬天、变成天鹅并找到归属",
    "vbench2-complex_plot-067": "收到未来来信、寻找古书和城堡、启动倒转的时钟并理解时间",
    "vbench2-complex_plot-080": "寻找红龙和金蛋、发现王国混乱、归还金蛋并恢复和平",
    "vbench2-complex_plot-072": "攀登火山、接受试炼、击败神秘敌人并保护岛屿",
}


CAMERA_ZH = {
    "wide establishing hold, a lateral follow through the handoff, and a centered support reveal":
        "宽幅建立镜头保持稳定，通过横向跟拍完成交接，最后以居中的支撑面揭示收尾",
    "gentle zoom toward the active object, a side-on follow that keeps both actors visible, and a held final frame":
        "朝活动物体缓慢变焦，使用保持两人可见的侧向跟拍，并停留在最终画面",
    "static opening composition, a motivated pan between lanes, and a quiet pull-back after placement":
        "以静态开场构图开始，在两条运动路线之间进行有动机的摇摄，放置后安静地拉远",
    "low tracking move for the first transfer, a short reverse arc for ownership change, and a stable finish":
        "第一次交接使用低机位跟踪移动，在所有权变化时使用短暂反向弧线，最后稳定收尾",
    "restrained push-in during the approach, a readable two-shot orbit at contact, and a slow final dolly":
        "接近过程中克制地推进，在接触时进行可读的双人环绕，最后缓慢推轨",
}


def _event_plan_zh(entity_plan: str) -> str:
    if "Alice carries the red cube to Bob" in entity_plan:
        return (
            "使用恰好两名指定人物 Alice 和 Bob，以及恰好两个代理物体：红色立方体和蓝色杯子。"
            "Alice 将红色立方体搬到 Bob 身边，然后把它交给 Bob；Bob 放下红色立方体。"
            "与此同时，Bob 将蓝色杯子搬到放置区并放下蓝色杯子。"
        )
    if "Bob carries the red cup to Dana" in entity_plan:
        return (
            "使用恰好两名指定人物 Bob 和 Dana，以及恰好两个代理物体：红色杯子和蓝色立方体。"
            "Bob 将红色杯子搬到 Dana 身边，然后把红色杯子交给 Dana。Dana 停顿后将红色杯子还给 Bob；"
            "Bob 放下红色杯子。与此同时，Dana 将蓝色立方体搬到放置区并放下蓝色立方体。"
        )
    if "Alice carries the green book while Dana carries the yellow ball" in entity_plan:
        return (
            "使用恰好两名指定人物 Alice 和 Dana，以及恰好两个代理物体：绿色书本和黄色球。"
            "Alice 搬运绿色书本，Dana 搬运黄色球；随后 Alice 将绿色书本交给 Dana。"
            "Dana 放下绿色书本，之后将黄色球放入放置区。"
        )
    if "Alice reveals the green ball" in entity_plan:
        return (
            "使用恰好两名指定人物 Alice 和 Carla，以及恰好两个代理物体：绿色球和黄色书本。"
            "Alice 先展示绿色球，然后将绿色球搬到 Carla 身边并交给 Carla。Carla 停顿后将绿色球还给 Alice，"
            "Alice 放下绿色球。之后，Carla 将已可见的黄色书本放入放置区。"
        )
    return f"实体与动作计划如下：{entity_plan}"


def translate_prompt(case_id: str, prompt: str) -> str:
    """Return a Chinese reviewer-facing rendering without changing ``prompt``."""

    scene = SCENE_ZH.get(case_id)
    if scene is None:
        return f"中文提示词翻译：暂未配置该案例的完整译文，请以以下原文为参考：{prompt}"

    source_match = re.search(r"The source dimension is (.+?) and its camera/action cue is (.+?)\. Use exactly ", prompt)
    entity_match = re.search(r"Use exactly (.+?) Schedule the events", prompt)
    camera_match = re.search(r" Use a (.+?); every active actor", prompt)
    if not source_match or not entity_match or not camera_match:
        return f"中文提示词翻译：{scene}"

    dimension, _cue = source_match.groups()
    entity_plan = entity_match.group(1)
    camera_plan = camera_match.group(1)
    dimension_zh = DIMENSION_ZH.get(dimension, dimension)
    cue_zh = CUE_ZH.get(case_id, "未指定")
    camera_zh = CAMERA_ZH.get(camera_plan, f"镜头安排：{camera_plan}")
    return (
        f"中文提示词翻译：VBench-2.0 种子上下文（不要添加未命名实体）：{scene} "
        f"原始维度为“{dimension_zh}”，镜头/动作提示为“{cue_zh}”。 "
        f"{_event_plan_zh(entity_plan)} "
        "按上述顺序将事件安排在连续的 6 秒镜头中：接触前加入预备动作；让所有权变化保持足够时间以便读清；"
        f"并保持最终支撑状态可见。镜头采用{camera_zh}；所有活动人物和物体都必须保持可识别，不得发生身份交换、"
        "无计划的路线交叉或相互穿插，也不得使用会遮挡交接的镜头切换。"
    )
