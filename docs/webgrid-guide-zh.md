# GazeFuse 实机 Webgrid 测试指南（中文）

本指南带你在 MacBook Pro 上完成：授予权限 → 校准 → 打开 Neuralink Webgrid → 测试 → 记录成绩和日志。全程大约 15 分钟。

## 0. 准备

- MacBook Pro（内置摄像头），光线均匀、脸部正面打光（不要背光坐在窗前）。
- 坐姿：眼睛离屏幕约 50–65 cm，屏幕在正前方，**整个测试过程尽量保持同样的坐姿**（校准是按坐姿做的）。
- 手：用来捏合（拇指+食指）的那只手要在摄像头画面里，放在键盘上方/胸前就行，手掌朝向摄像头侧面也可以。
- AirPods 3：戴好并连接到这台 Mac（可选，用于头部微调和咬牙点击）。
- 浏览器：Chrome 或 Safari，网页缩放 100%。

## 1. 构建并安装

```bash
cd ~/myproj/gazefuse
scripts/bundle.sh --install        # 生成并安装 ~/Applications/GazeFuse.app
open ~/Applications/GazeFuse.app
```

菜单栏右上角会出现 **◉ GF** 图标。

## 2. 授予权限（只需一次）

1. **摄像头**：首次启动会弹窗，点"允许"。如果错过了：菜单 → *Open Camera Settings*，打开 GazeFuse。
2. **辅助功能（Accessibility）**：用于移动光标和点击（CGEvent）。菜单 → *Open Accessibility Settings*，把 GazeFuse 打开。
   - 注意：每次重新编译（ad-hoc 签名会变）后，可能要在这里把 GazeFuse **关掉再打开**一次，或删掉后重新添加。
3. **运动与健身（Motion）**：第一次连上 AirPods 时会弹窗，点"允许"（用于读取 AirPods 头部姿态）。
4. 权限改完后，菜单 → *Quit GazeFuse*，再重新打开一次。

打开菜单，第一行状态应显示类似：`cam 30 fps · gaze ✓ · hand ✓ · AirPods ✓ · AX ✓ · cal ✗`。

## 3. 校准（每次换坐姿/光线都建议重做）

1. 菜单 → *Calibrate Gaze…*（或 ⌃⌥C）。
2. 屏幕变黑，依次出现圆点，**头保持不动，只用眼睛盯住每个圆点的中心**，直到它缩小消失。
3. 之后会有几个验证点，最后弹窗显示平均误差（单位 pt）。参考：< 60 pt 较好，> 120 pt 建议重新校准（检查光线、坐姿、眼镜反光）。
4. 按 Esc 可随时取消。校准结果会自动保存，下次启动自动加载。

建议打开 *Show / Hide HUD*（⌃⌥H）：蓝色圆圈 = 原始视线估计，绿色点 = 融合后的光标，方便你直观看到精度。

## 4. 操作方式

| 动作 | 效果 |
| --- | --- |
| 眼睛看向目标 | 光标快速跳到视线附近（粗定位） |
| 头部轻微转动（戴 AirPods） | 光标细调（精定位），像"用鼻尖推光标" |
| 拇指+食指捏一下 | 左键单击（捏住不放 = 拖拽） |
| 拇指+中指捏一下 | 右键单击 |
| 捏住后移动手 | 细调光标 / 拖拽 |
| 用力咬一下牙（戴 AirPods） | 备用左键单击 |
| 碰一下真实鼠标/触控板 | 自动暂停视线控制 1 秒，不会抢鼠标 |
| ⌃⌥G | 开/关光标控制（紧急停止也用它） |

先在桌面上练习 2–3 分钟，再开始正式测试。

## 5. 跑 Webgrid

1. 菜单 → *Start Recording*（⌃⌥R），开始记录原始传感器数据（可离线复现/调参）。
2. 菜单 → *Open Neuralink Webgrid*，浏览器打开 https://neuralink.com/webgrid/ ，把浏览器窗口最大化或全屏。
3. 选择网格大小（默认 30×30；想先找手感可以先用较小的网格，但**正式成绩请记下网格大小**）。点击类型只用左键（我们的 BPS 按单一点击类型计算最公平；若你用了多种点击类型也请记下）。
4. 按 ⌃⌥G 开启光标控制。菜单 → *Mark: Webgrid run start*。
5. 用 眼睛 + 捏合 去点亮的蓝色格子，尽量快、尽量少点错。一局结束后，Webgrid 页面会显示 **BPS / NTPM** 等成绩。
6. 菜单 → *Mark: Webgrid run end*，再按 ⌃⌥G 关闭光标控制，⌃⌥R 停止录制。

## 6. 记录成绩

- **截图**：成绩页面出现后按 ⌘⇧4 截取成绩区域（或 ⌘⇧3 全屏），截图默认保存在桌面。
- 在 `docs/live-results.md`（或发给我）记下：日期、网格大小、点击类型数、BPS、NTPM、使用的通道（gaze / pinch / AirPods head / jaw）、校准误差。
- 录制文件在：菜单 → *Open Data Folder* → `recordings/session-*.jsonl`。离线复现：
  ```bash
  swift run -c release gazefuse-bench replay ~/Library/Application\ Support/GazeFuse/recordings/session-XXXX.jsonl
  ```
  会输出两次 mark 之间的点击次数、光标统计，用来对照网页成绩并调参。
- BPS 计算公式（与 Webgrid 前端一致）：`BPS = max(log2(N − 1) × NTPM / 60, 0)`，N = 格子数（×点击类型数），NTPM = 每分钟净正确数（正确 − 错误）。

## 7. 建议的测试矩阵（每项 2–3 局取最好）

1. 鼠标/触控板（基准）
2. 仅视线 + 捏合点击（关闭 AirPods 两个通道）
3. 视线 + 捏合 + AirPods 头部微调
4. 视线 + 咬牙点击 + AirPods 头部微调（关闭手部通道）

在菜单 *Input Channels* 里勾选/取消对应通道。

## 常见问题

- **光标乱跳**：重新校准；检查背光、眼镜反光；保持头部位置与校准时一致。
- **捏合没反应**：手离摄像头太近/太远或出画；HUD 中 hand 应显示 on。
- **AirPods 没数据**：确认 AirPods 已连到这台 Mac 且戴在耳朵上，系统设置里允许了"运动与健身"。
- **光标不动**：辅助功能权限没生效 → 在辅助功能里把 GazeFuse 关掉再打开，然后重启 App。
