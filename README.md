# 低延迟自动答题器

程序从 Windows 桌面上的 scrcpy 视频区域抓取同一帧，批量识别题干与四个选项，
优先用保守规则求解，无法确定时请求本地 Ollama，最后才通过 ADB 点击。
空 OCR、低置信度、非法答案或页面变化都会阻止点击；暂时性识别失败会自动重试，
持续失败才保存一次调试资料并回到题目页等待，不会终止主循环。

## 配置

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

离线演练仍会加载 PaddleOCR；只有规则无法判断时才会请求 Ollama。

CPU OCR 默认设置 `ocr.enable_mkldnn = false`，用于规避部分 PaddlePaddle 3.x
版本在 oneDNN/PIR 指令转换中的运行错误。关闭后可能略慢，但不会影响识别逻辑。
内容识别默认使用 `PP-OCRv5_mobile_det` 和 `PP-OCRv5_mobile_rec`，相较 medium
模型更适合实时答题；模型名称可通过 `ocr.text_detection_model_name` 和
`ocr.text_recognition_model_name` 调整。

运行时只有连续检测到相同题号、四个白框和稳定文字区域才会 OCR。点击后必须观察到
画面变化、连续两次识别出不同的新题号、白框恢复并稳定，才会处理下一题。结算页、
弹窗或长时间无题目特征只会让程序回到等待状态。

Ollama 默认在程序启动时后台预热，不阻塞题目页检测。每道题都会输出如下性能日志：

```text
TIMING question=1 page_confirm_ms=431 ocr_ms=1755 solve_ms=0 recognize_to_decision_ms=1756
```

实时点击还会输出 `tap_ms` 和 `recognize_to_tap_ms`，用于区分页面检测、OCR、求解及
ADB 点击各阶段耗时。

## 失败资料与日志

默认日志写入 `artifacts/auto-answer.log`。失败目录包含可获得的完整帧、题干和四个
选项 ROI、`ocr.json`、统一题目文本以及 `error.txt`。程序不会在不确定时随机点击。

## 开发检查

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest
```
