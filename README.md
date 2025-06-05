# MADDPG 多无人机协同系统 (3v3 Qt5 模拟环境版)

本项目实现了基于 MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 算法的多无人机协同搜索与围捕系统。当前版本在 Python 中模拟了一个 3v3 的对抗环境，该环境旨在模拟未来与 Qt5 集成的诸多特性，如传感器范围、信息共享和特定的敌我行为。

## 环境依赖

### Python 环境 (训练与模拟部分)

- Python: 3.9.0+
- numpy: 1.26.0+
- gymnasium: 0.26.2+ (或 gym)
- pillow: 10.2.0+ (用于图像处理，例如在评估时生成 GIF)
- torch: 2.4.0+ (PyTorch)
- matplotlib: 用于可视化学习曲线和评估

### Qt 环境 (未来集成目标)

- Qt5.12+ (推荐 Qt5.15)
- C++11 及以上

## 项目结构

### Python 训练与模拟模块

- `maddpg.py`: MADDPG 算法核心逻辑，包含 `MADDPG` 类，管理多个 `Agent`。
- `networks.py`: Actor 和 Critic 网络的 PyTorch 定义。
- `buffer.py`: 多智能体经验回放缓冲区 (`MultiAgentReplayBuffer`)。
- `qt5_sim_env.py`: 核心的 3v3 无人机模拟环境 (`Qt5SimUAVEnv`)，包含复杂的交互逻辑、状态表示和奖励计算。
- `main_qt5_sim.py`: 用于启动和管理 MADDPG 算法在`Qt5SimUAVEnv`上的训练过程。
- `evaluate_qt5_sim.py`: 用于加载训练好的模型，在模拟环境中进行评估并可视化结果（例如生成 GIF 动画）。
- `UAV.png`, `Enemy_UAV.png`: 可选的无人机图标，用于渲染（如果评估脚本中包含可视化渲染）。

### Qt 集成模块 (与 Python 模型交互)

- `maddpg_agent.h`/`maddpg_agent.cpp`: 定义了 `UAVModel` C++类，用于在 Qt 项目中加载和运行由 Python 训练导出的单个无人机 Actor 网络的控制模型。
- `[scenario_name]_agent_X_actor_inference.json`: 训练好的单个 UAV 的 Actor 网络参数文件 (X 为智能体编号，如 0, 1, 2)，以 JSON 格式存储，供 `UAVModel` 加载。

## `Qt5SimUAVEnv` 环境详解

### 1. 仿真参数 (部分参数可在 `qt5_sim_env.py` 构造函数中调整):

- **地图尺寸**: 默认为 1280x800 像素。
- **时间步长**: 默认为 0.1 秒。
- **我方无人机 (Agent UAVs)**:
  - 数量: 3
  - 初始位置: 地图右下角区域，随机分布。
  - 最大速度分量: 50.0 像素/秒 (各轴)。
  - 初始健康值: 100。
  - 探测半径: 300.0 像素 (用于探测敌机和障碍物)。
  - 攻击半径: 150.0 像素。
  - 近似碰撞半径: 10.0 像素。
- **敌方无人机 (Enemy UAVs)**:
  - 数量: 3
  - 初始位置: 地图左上角区域，随机分布。
  - 移动速度: 50.0 像素/秒。
  - 初始健康值: 100。
  - 行为: (当前版本)
    - 被探测到后，会朝远离最近的我方 UAV 的方向逃跑。
    - 未被探测到时，以当前速度和方向飞行，有较小几率随机改变方向。
  - 近似碰撞半径: 10.0 像素。
- **障碍物**:
  - 静态障碍物数量: 2
  - 动态障碍物数量: 1 (最大速度 20.0 像素/秒)
  - 总障碍物数量: 3
  - 半径: 固定为 80.0 像素
  - 随机生成位置（避开 UAV 出生区域）。

### 2. 观察空间 (Observation Space) - 每个 Agent 31 维向量:

每个我方 UAV 接收一个 31 维的浮点数向量作为观察，所有值都经过归一化处理。

- **自身信息 (5 维)**:
  - `[norm_pos_x, norm_pos_y, norm_vel_x, norm_vel_y, norm_health]`
  - 位置归一化到 `[0,1]` (除以地图宽高)，速度归一化到 `[-1,1]` (除以最大速度分量)，健康值归一化到 `[0,1]` (除以 100)。
- **队友信息 (6 维 = 2 队友 × 3 维)**:
  - 对每个队友: `[norm_rel_pos_x, norm_rel_pos_y, norm_health]`
  - 相对位置归一化到 `[-1,1]` (除以地图宽高)。如果队友被击毁，其信息用 `[0,0,0]` 填充。
- **敌机信息 (9 维 = 3 敌机 × 3 维)**:
  - 对每个敌机: `[norm_rel_pos_x, norm_rel_pos_y, norm_health]`
  - 仅包含被当前 UAV 传感器探测到的、且存活的敌机信息。按距离排序，优先报告近的。
  - 如果敌机未被探测到、或已被击毁，其对应槽位用 `[0,0,0]` 填充。
- **搜索目标信息 (2 维)**:
  - `[norm_rel_search_target_x, norm_rel_search_target_y]`
  - 仅在搜索模式 (`self.search_mode == True`) 下有效。表示当前 UAV 应前往的搜索区域中心点的相对归一化位置。
  - 在围捕模式下，此部分为 `[0,0]`。
- **障碍物信息 (9 维 = 3 障碍物 × 3 维)**:
  - 对视野内探测到的最多 3 个障碍物（按与无人机表面的距离排序）: `[norm_rel_pos_x, norm_rel_pos_y, norm_radius]`
  - 相对位置归一化，半径使用地图宽高中的最大值归一化。
  - 如果视野内可观测的障碍物少于 3 个，则用 `[0,0,0]` 填充剩余槽位。

### 3. 动作空间 (Action Space) - 每个 Agent 2 维向量:

- 每个 UAV 的动作是一个 2 维连续向量 `[a_x, a_y]`，每个分量范围 `[-1, 1]`。
- 该向量代表归一化的加速度指令，在环境内部会乘以一个缩放因子 (`uav_input_accel_scale = 50.0` 默认) 得到实际的加速度值 (像素/秒 ²)。

### 4. 奖励函数 (`_calculate_rewards`):

每个 UAV 在每一步都会收到一个奖励值，该奖励综合了多种因素（详细设计见 `qt5_sim_env.py`）。关键的奖励和惩罚包括：

- 生存/死亡惩罚。
- 接近搜索目标奖励。
- 追踪和攻击敌机奖励（包括最佳追踪距离、进入攻击范围等）。
- 团队协作攻击奖励。
- 击杀敌机奖励。
- 碰撞障碍物惩罚。
- 任务完成（所有敌机被击毁）的全局奖励。

**注意**: 当前版本的奖励函数不包含"上帝视角辅助搜索奖励"。搜索目标由 `_get_search_target` 方法基于区域划分生成。

## 训练与评估

### 1. 设置 Python 环境:

```bash
# (可选) 创建并激活虚拟环境
# python -m venv .venv
# source .venv/bin/activate  # Linux/macOS
# .\\.venv\\Scripts\\activate # Windows

# 安装依赖
pip install numpy gymnasium torch matplotlib pillow
# 或者如果有 requirements.txt: pip install -r requirements.txt
```

### 2. 运行训练:

```bash
cd MADDPG_Multi_UAV_Roundup
python main_qt5_sim.py
```

- 训练参数 (如总轮数、学习率、网络结构等) 可在 `main_qt5_sim.py` 中调整。
- 模型检查点 (PyTorch 的 `.pth` 文件和用于 C++ 推理的 `.json` 文件) 和学习曲线图片将保存在 `tmp/maddpg_qt5/[scenario_name]/` 目录下。
- `scenario_name` 默认为 `UAV_Qt5_3v3_SensorCircle`，可在 `main_qt5_sim.py` 中修改。

### 3. 运行评估:

```bash
python evaluate_qt5_sim.py
```

- 评估脚本会加载在上述训练输出目录中找到的最新或最佳模型 (通常是 `[scenario_name]_agent_X_actor.pth` 和对应的 `[scenario_name]_agent_X_actor_inference.json`)。
- 它将运行若干个评估回合，输出性能统计数据 (如胜率、平均得分、平均搜索/围捕时间、击毁敌机数)，并可能生成一个 `evaluation_rollout.gif` 来可视化一个回合的执行情况。

## Qt 项目集成指南

`maddpg_agent.h` 和 `maddpg_agent.cpp` 文件提供了一个 `UAVModel` 类，用于在 Qt C++项目中加载和使用 Python 训练的 Actor 模型进行推理。

### 主要步骤:

1.  **训练模型**: 使用 `main_qt5_sim.py` 训练模型。对于每个智能体 (Agent)，这将生成 PyTorch 模型文件 (例如 `UAV_Qt5_3v3_SensorCircle_agent_0_actor.pth`) 和对应的 JSON 推理参数文件 (例如 `UAV_Qt5_3v3_SensorCircle_agent_0_actor_inference.json`)。

2.  **在 Qt 中为每个我方 UAV 加载其对应的模型**:

    ```cpp
    #include "maddpg_agent.h" // 确保路径正确
    // ... 其他Qt包含 ...

    // 在您的类中 (例如 MainWindow 或一个专门的UAV控制类)
    // 假设场景名为 "UAV_Qt5_3v3_SensorCircle" (与训练时一致)
    // 基础路径到模型保存目录
    QString baseModelPath = "path/to/your/workspace/MADDPG_Multi_UAV_Roundup/tmp/maddpg_qt5/UAV_Qt5_3v3_SensorCircle/";
    QString scenarioName = "UAV_Qt5_3v3_SensorCircle"; // 确保与训练时一致

    UAVModel* uav0_brain = new UAVModel(0, this); // Agent 0
    QString modelPathAgent0 = baseModelPath + scenarioName + "_agent_0_actor_inference.json";
    bool loaded0 = uav0_brain->loadModel(modelPathAgent0);
    if (!loaded0) { qDebug() << "Agent 0 model load failed for: " << modelPathAgent0; }
    else { qDebug() << "Agent 0 model loaded successfully from: " << modelPathAgent0; }

    UAVModel* uav1_brain = new UAVModel(1, this); // Agent 1
    QString modelPathAgent1 = baseModelPath + scenarioName + "_agent_1_actor_inference.json";
    bool loaded1 = uav1_brain->loadModel(modelPathAgent1);
    if (!loaded1) { qDebug() << "Agent 1 model load failed for: " << modelPathAgent1; }
    else { qDebug() << "Agent 1 model loaded successfully from: " << modelPathAgent1; }

    UAVModel* uav2_brain = new UAVModel(2, this); // Agent 2
    QString modelPathAgent2 = baseModelPath + scenarioName + "_agent_2_actor_inference.json";
    bool loaded2 = uav2_brain->loadModel(modelPathAgent2);
    if (!loaded2) { qDebug() << "Agent 2 model load failed for: " << modelPathAgent2; }
    else { qDebug() << "Agent 2 model loaded successfully from: " << modelPathAgent2; }
    ```

    **注意**: `main_qt5_sim.py` 中定义的 `scenario_name` 会作为模型文件路径和 JSON 文件名的一部分。你需要根据实际的 `scenario_name` 和智能体索引 (`agent_0`, `agent_1`, `agent_2`) 来构造正确的文件路径。

3.  **准备观察向量**:
    从您的 Qt 仿真环境或实际传感器收到的数据，按照本项目 "观察空间" 部分描述的 31 维格式为每个 UAV 组装其 `QVector<float>` 格式的观察向量。所有值必须与 Python 训练时采用相同的归一化方法。

    ```cpp
    // 为 UAV 0 准备观察向量 (示例)
    QVector<float> current_observation_for_uav0;
    // --- 仔细按照README中描述的31维格式和顺序填充观察数据 ---
    // 示例:
    // 自身 (5)
    current_observation_for_uav0 << 0.8f /*x*/ << 0.8f /*y*/ << 0.0f /*vx*/ << 0.0f /*vy*/ << 1.0f /*health*/;
    // 队友1 (3) (假设是 UAV 1 相对于 UAV 0)
    current_observation_for_uav0 << -0.1f /*rel_x*/ << 0.05f /*rel_y*/ << 1.0f /*health*/;
    // 队友2 (3) (假设是 UAV 2 相对于 UAV 0)
    current_observation_for_uav0 << 0.05f /*rel_x*/ << -0.1f /*rel_y*/ << 1.0f /*health*/;
    // 敌机 1 (3) - 假设未探测到
    current_observation_for_uav0 << 0.0f << 0.0f << 0.0f;
    // 敌机 2 (3) - 假设探测到
    current_observation_for_uav0 << -0.3f /*rel_x*/ << -0.4f /*rel_y*/ << 0.8f /*health*/;
    // 敌机 3 (3) - 假设未探测到
    current_observation_for_uav0 << 0.0f << 0.0f << 0.0f;
    // 搜索目标 (2) - 假设 UAV 0 在围捕模式
    current_observation_for_uav0 << 0.0f << 0.0f;
    // 障碍物 1 (3)
    current_observation_for_uav0 << -0.05f /*rel_x*/ << -0.02f /*rel_y*/ << 0.03f /*norm_radius*/;
    // 障碍物 2 (3)
    current_observation_for_uav0 << 0.1f /*rel_x*/ << -0.08f /*rel_y*/ << 0.025f /*norm_radius*/;
    // 障碍物 3 (3) - 假设只探测到2个，第3个用0填充
    current_observation_for_uav0 << 0.0f << 0.0f << 0.0f;


    if (current_observation_for_uav0.size() != 31) { // 严格检查维度
        qWarning() << "观察向量维度不正确! 期望31, 实际: " << current_observation_for_uav0.size();
        // 处理错误...
    }
    ```

4.  **获取并应用动作**:

    ```cpp
    // 假设 uav0_brain 已经为 Agent 0 成功加载了模型
    if (uav0_brain && loaded0) { // 确保对象存在且模型已加载
        QPointF normalized_action_agent0 = uav0_brain->inference(current_observation_for_uav0);
        qDebug() << "UAV 0 action: ax=" << normalized_action_agent0.x() << ", ay=" << normalized_action_agent0.y();

        // 将此归一化动作应用到 Qt 环境中的无人机 0 的动力学模型
        // float actual_accel_x = normalized_action_agent0.x() * uav_input_accel_scale_in_qt;
        // float actual_accel_y = normalized_action_agent0.y() * uav_input_accel_scale_in_qt;
        // ... (应用到物理引擎或发送控制指令) ...
    }
    // 对 uav1_brain, uav2_brain 重复此过程 (使用它们各自的观察向量)
    ```

    **重要**: `UAVModel::inference` 返回的是归一化动作 `[a_x, a_y]`，每个分量在 `[-1, 1]` 范围内。您需要在 Qt 端根据您的无人机控制方式将其转换为实际的物理量（例如，具体的加速度值）。Python 环境 `qt5_sim_env.py` 中的 `uav_input_accel_scale` (默认为 50.0 像素/秒 ²) 是一个重要参考，它用于将归一化动作映射到 Python 仿真环境中的加速度值。您在 C++ 端应采用类似或兼容的缩放和解释机制。

## 许可证

MIT License

## 致谢

- 本项目的 MADDPG 实现部分参考了 [Phil Tabor 的多智能体强化学习代码](https://github.com/philtabor/Multi-Agent-Reinforcement-Learning)。
