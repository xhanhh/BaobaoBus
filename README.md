# 宝宝巴士巅峰对决 自动化答题程序

该程序可以以OCR从屏幕上获取宝宝巴士pk题目和选项，先匹配一波本地规则集，未命中则请求OpenAI（目前用的阿里千问的提示词）或者Ollama接口获取题目答案，使用安卓adb触控。

## 环境需求

- Windows 10+
- Python 3.13 (venv) （需要版本一致）
- PaddlePaddleOCR(GPU): 英伟达显卡并安装Cuda12.9（Cuda版本必须严格一致）
- Android ADB、Scrcpy: 可以启用开发者选项和ADB调试的安卓手机（最好A14+）
- LLM: Ollama或者阿里云大模型百炼（OpenAI）

## 其他的注意事项

分辨率和坐标有三个：

1. 手机原本的分辨率坐标（主要用于adb触摸）
2. scrcpy投屏到电脑上的分辨率坐标
3. 电脑屏幕的分辨率坐标（按配置应该与上面的坐标一样，用于ocr与检测）

默认配置是用的小米14调整的，手机不同`launch-scrcpy.ps1`里分辨率比例和adb坐标也不一定一样，需要自行调整。
电脑屏幕ocr的坐标肯定更不一样了，这个需要慢慢来。

## 声明

- 本项目使用Codex进行辅助开发。
- 该软件没有对任何官方软件、客户端进行修改。只是模拟人为行为。

### 使用条款

一旦您使用本软件，代表您接受了以下使用条款。

- 本项目一切开发旨在学习大模型的应用，请勿用于非法用途。若造成不良影响后果的，由使用者独自承担全部责任。
- 本项目开源，以AGPLv3协议授权，其他衍生版本不受我们控制。
- 使用本软件意味着您自愿承担使用该软件可能所带来的风险与后果，包括但不限于账号封禁等。我们也不保证您使用本软件不会遭受惩罚（如封禁等）。
- 用户使用该软件所做出的行为与我们无关。
- 若该软件侵犯了您的合法权益，请及时与我们联系。

## 配置

下面都是Codex写的文档了，大致应该能看明白。

先复制示例：

```powershell
Copy-Item -LiteralPath config.example.toml -Destination config.toml
```

必须手动校准两套相互独立的坐标：

1. `capture` 是 scrcpy 视频在 Windows 虚拟桌面上的绝对截图范围。
2. `regions.question_number` 只包含顶部“第 X 题”文字。
3. `regions.question` 和四个 `regions.options` 是题干、选项文字及稳定检测 ROI。
4. 四个 `regions.option_boxes` 覆盖完整白色选项框，仅用于确认题目页。
5. 四个 `adb.tap_points` 是手机原始屏幕坐标，必须用 `adb shell wm size` 和实际点击
   位置校准。它们绝不能从投屏 ROI 坐标直接照搬。

选项 ROI 无论在 TOML 中如何排列，都会按从上到下、同一行从左到右编号为
`0、1、2、3`。示例 ROI 只对应 `res/example-screenshot.jpg`，不代表当前 scrcpy
窗口或手机坐标。

先检查配置结构（允许 ADB 点位仍是 `-1`）：

```powershell
.\.venv\Scripts\python.exe -m auto_answer --config config.toml --check-config
```

## 启动

依赖全部记录在 `pyproject.toml`：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

先启动并固定 scrcpy 窗口，再运行：

```powershell
.\tools\launch-scrcpy.ps1
.\.venv\Scripts\python.exe -m auto_answer --config config.toml
```

用 Ctrl+C 安全停止。限制处理题数可加 `--max-questions 10`。

可用示例截图做不点击手机的单题演练：

```powershell
.\.venv\Scripts\python.exe -m auto_answer --config config.example.toml `
  --image res\example-screenshot.jpg --dry-run
```

离线演练仍会加载 PaddleOCR；只有规则无法判断时才会请求已配置的 LLM。

### LLM 来源

`llm.provider_order` 决定模型调用顺序。例如阿里云优先、本地兜底：

```toml
[llm]
provider_order = ["aliyun", "ollama"]

[openai_compatible]
enabled = true
base_url = "https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
model = "qwen3.7-flash" # 也可改为 qwen3.7-plus
api_key = "sk-..."
api_key_env = "DASHSCOPE_API_KEY"
enable_thinking = false
```
可以手动在config中配置api-key，也可以用环境变量：

```powershell
$env:DASHSCOPE_API_KEY = "sk-..."
```

阿里云请求使用 OpenAI 兼容的 `/chat/completions`、`response_format = json_object`、
`temperature = 0` 并关闭 thinking；响应仍经过
与 Ollama 相同的答案序号、算式、答案值和选项对应关系校验。首选来源出现 HTTP
错误、超时、空响应或严格校验失败时，才会尝试下一个来源。

内容识别默认使用 `PP-OCRv5_mobile_det` 和 `PP-OCRv5_mobile_rec`，相较 medium
模型更适合实时答题；模型名称可通过 `ocr.text_detection_model_name` 和
`ocr.text_recognition_model_name` 调整。

运行时只有连续检测到相同题号、四个白框和稳定文字区域才会 OCR。点击后必须观察到
画面变化、连续两次识别出不同的新题号、白框恢复并稳定，才会处理下一题。结算页、
弹窗或长时间无题目特征只会让程序回到等待状态。

### 挑战结束后自动继续

设置 `post_challenge.enabled = true` 后，最后一题点击且下一题等待超时，程序会：

1. 连续确认绿色挑战成功横幅或灰蓝挑战失败横幅，同时确认橙色“继续挑战”按钮；
2. 挑战成功时等待可能延迟出现的“排名提升啦”弹层；
3. 若弹层出现，连续确认粉色面板和关闭按钮后点击叉号；
4. 连续确认弹层消失、挑战成功页重新变亮，再点击“继续挑战”；
5. 挑战失败时不等待排名弹层，稳定确认失败页后直接点击“继续挑战”；
6. 等待并连续确认 Ready/Go，随后交回正常首题检测流程。

`success_banner`、`continue_button`、`ranking_panel` 和
`ranking_close_button` 都是相对投屏截图的 ROI。`ranking_close_tap` 和
`continue_challenge_tap` 则是手机原始屏幕坐标，两套坐标不能混用。示例值按
1500×674 投屏画面和 2670×1200 手机画面从录屏标定；窗口缩放或手机分辨率改变后
必须分别重新校准。该流程使用多区域颜色特征并要求连续三帧成立，不使用单一颜色
直接触发点击。

Ollama 默认在程序启动时后台预热，不阻塞题目页检测。每道题都会输出如下性能日志：

```text
TIMING question=1 page_confirm_ms=431 ocr_ms=1755 solve_ms=0 recognize_to_decision_ms=1756
```

首题还会输出：

```text
FIRST_PAGE_TIMING ready_to_answer_ms=2920 answer_layout_to_confirm_ms=398 number_source=ready-inferred
```

`page_confirm_ms` 包含匹配、VS、Ready/Go 和题目淡入动画，不能单独用来判断检测延迟。
`ready_to_answer_ms` 是 Ready 到答题页开始淡入的游戏动画耗时；
`answer_layout_to_confirm_ms` 才是答题布局首次可见到程序确认的耗时。程序在
`regions.ready_indicator` 检测到 Ready/Go 后，会在
`state.ready_fast_window_seconds` 内改用 `state.ready_poll_interval_seconds`
高频检测。Ready 后的首题号可以配置为推断 1，但完整 OCR 仍会再次识别并校验题号，
校验失败时不会点击。

`state.infer_sequential_question_number = true` 对后续题采用同样的安全优化：
页面已明显变化、四个选项框完整且内容稳定后先推断题号为 `N+1`，省去一次独立题号
OCR；紧接着的批量 OCR 仍必须识别出相同题号，否则本轮结果作废且不会点击。

实时点击还会输出 `tap_ms` 和 `recognize_to_tap_ms`，用于区分页面检测、OCR、求解及
ADB 点击各阶段耗时。

纯数字选项首先使用带算式和值校验的结构化响应。如果响应为空、超时、格式错误或
算式与答案矛盾，各来源的 `retry_numeric_as_text = true` 会让程序再请求一次仅返回
选项序号的 JSON 模式；该来源仍失败后才切换到下一个来源。

`fallback.random_on_ocr_failure` 和 `fallback.random_on_llm_failure` 默认均为
`false`，此时程序不会点击当前题，等待用户人工处理并进入下一题。将对应选项设为
`true` 会在重新确认题目页稳定后随机点击 `0～3`，这种模式明确可能答错。

`state.overlap_ocr_with_stability = true` 会提前一个确认帧开始 OCR，并在 OCR
结束后用新帧复核内容；如果淡入动画仍在变化，本次结果会被丢弃并按原流程重试。

`state.title_min_white_ratio` 会先用题号 ROI 中的亮色像素比例判断“第 X 题”是否
可见。淡出期间即使四个白框暂时仍在，也会跳过重量级题号 OCR；每隔
`state.title_probe_interval_seconds` 仍会强制探测一次，避免阈值不合适时永久等待。
默认阈值按当前录屏测得，调整题号 ROI、缩放或画质后应重新校准。

`adb.persistent_shell = true` 会预先建立长期 ADB shell，避免每题点击都创建新进程。
长连接超时或断开时，该次点击会自动改用原来的单次 ADB 命令，随后重建长连接。

## 失败资料与日志

默认日志写入 `artifacts/auto-answer.log`。失败目录包含可获得的完整帧、题干和四个
选项 ROI、`ocr.json`、统一题目文本以及 `error.txt`。只有显式开启相应的
`fallback.random_on_*` 配置时，程序才会在最终失败后随机点击。

## 代码结构

- `core`：配置、领域模型和异常。
- `vision`：截图、OCR、文本规范化和页面状态检测。
- `device`：ADB 设备控制。
- `solving`：LLM 路由、Ollama、OpenAI 兼容客户端与本地规则。
- `solving/rules`：按算术、应用题、金额、计数和数概念拆分的规则模块。
- `runtime`：中央调度器和失败材料记录。

## 开发检查

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest
```
