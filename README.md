## Key dependencies based (Lower versions are not guaranteed to be feasible):

python: 3.9.0

numpy: 1.26.0

gym: 0.26.2

pillow: 10.2.0

torch: 2.4.0+cu124

torchaudio: 2.4.0+cu124

torchvision: 0.19.0+cu124

## Explanation of Document

- `agent`/`buffer`/`maddpg`/`networks`: Refer to Phil's work -> [PhilMADDPG](https://github.com/philtabor/Multi-Agent-Reinforcement-Learning);

- `sim_env`: Customized Multi-UAV round-up environment;

- `main`: Main loop to train agents;

- `main_evaluate`: Only rendering part is retained in `main`, in order to evaluate models (a set of feasible models is provided in `tmp/maddpg/UAV_Round_up`;

- `math_tool`: some math-relevant functions.

<img title="" src="Roundup.png" alt="" data-align="center" width="470">

---

## Stargazers over time
[![Stargazers over time](https://starchart.cc/reinshift/MADDPG_Multi_UAV_Roundup.svg?variant=adaptive)](https://starchart.cc/reinshift/MADDPG_Multi_UAV_Roundup)




#####
## 如何使用模型进行推理

### 在Qt项目中使用UAVModel类

```cpp
// 在您的MainWindow或其他类中
#include "uavmodel.h"
#include <QMqttClient>

// 创建模型实例（指定无人机ID）
UAVModel *uavModel = new UAVModel(0, this); // 0是无人机ID

// 加载模型参数
bool success = uavModel->loadModel("e:/Qt/Qt_project/MqttClient/multiUAVs/MADDPG_Multi_UAV_Roundup/agent_0_actor_inference.json");
if (success) {
    qDebug() << "模型加载成功";
    
    // 准备观察数据（26维向量）
    QVector<float> observation = {
        // 1-2: 自身位置 [x, y]
        0.5f, 0.3f,
        
        // 3-4: 自身速度 [vx, vy]
        0.01f, 0.02f,
        
        // 5-12: 团队成员相对位置 [dx1, dy1, dx2, dy2, dx3, dy3, ...]
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        
        // 13-14: 目标相对距离和角度 [distance, angle]
        0.8f, 0.5f,
        
        // 15-26: 激光传感器数据（12个传感器）
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };
    
    // 执行推理获取速度向量
    QPointF velocity = uavModel->getVelocity(observation);
    
    qDebug() << "推理结果 - 速度向量:" << velocity;
    
    // 使用速度向量控制无人机
    // 例如，通过MQTT发送到SylixOS
    QMqttClient *mqttClient = new QMqttClient(this);
    mqttClient->setHostname("192.168.1.100");
    mqttClient->setPort(1883);
    mqttClient->connectToHost();
    
    // 构造JSON消息
    QString message = QString("{\"uav_id\":%1,\"vx\":%2,\"vy\":%3}")
                      .arg(uavModel->uavId)
                      .arg(velocity.x())
                      .arg(velocity.y());
    
    // 发布消息
    mqttClient->publish("uav/control", message.toUtf8());
}
```
####