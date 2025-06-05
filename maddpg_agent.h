#ifndef MADDPG_AGENT_H
#define MADDPG_AGENT_H

#include <QObject>
#include <QVector>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QDebug>
#include <QtMath>
#include <QPointF>

/**
 * @brief MADDPG 单个 Actor 网络推理模型类 (C++/Qt)
 *
 * 该类用于加载由 Python (使用 PyTorch 和 MADDPG 算法) 训练好的单个无人机 (Agent)
 * 的 Actor 网络模型参数 (通常以 JSON 格式导出)。
 * 加载后，它可以根据输入的观察向量 (observation) 执行前向传播计算，
 * 输出归一化的动作指令，供无人机在 Qt 仿真环境或实际应用中进行控制决策。
 *
 * 主要设计用于与 MADDPG_Multi_UAV_Roundup 项目中的 Python 训练脚本配合使用。
 */
class UAVModel : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param id 无人机/智能体 ID，用于区分和加载对应的模型文件。
     * @param parent 父对象指针
     */
    explicit UAVModel(int id = 0, QObject *parent = nullptr);

    /**
     * @brief 加载 Actor 模型参数
     * @param filePath 模型参数 JSON 文件路径 (例如 "agent_0_actor_inference.json")
     * @return 是否加载成功
     */
    bool loadModel(const QString &filePath);

    /**
     * @brief 执行 Actor 网络推理计算, 获取归一化动作
     * @param observation 观察向量 (应与 Python 训练环境中的维度和归一化方式一致)
     * @return 动作向量 [a_x, a_y] (通常为归一化的加速度或速度指令, 各分量在 [-1, 1] 之间)
     */
    QPointF inference(const QVector<float> &observation);

    /**
     * @brief 获取目标速度 (基于模型输出的加速度方向和最大速度)
     * @param observation 观察向量
     * @return 目标速度 [vx, vy]，方向由模型输出决定，大小为 vMax
     */
    QPointF getTargetVelocity(const QVector<float> &observation);

    /**
     * @brief 获取更新后的速度 (考虑模型输出的加速度、当前速度和时间步长)
     * @param observation 观察向量
     * @return 更新后的速度 [vx, vy]，已考虑 vMax 限制
     */
    QPointF getVelocity(const QVector<float> &observation);

    /**
     * @brief 设置最大速度
     * @param maxVelocity 最大速度值
     */
    void setMaxVelocity(float maxVelocity) { vMax = maxVelocity; }

    /**
     * @brief 设置时间步长
     * @param dt 时间步长（秒）
     */
    void setTimeStep(float dt) { timeStep = dt; }

    /**
     * @brief 重置当前速度
     */
    void resetVelocity() { currentVelocity = QPointF(0, 0); }

private:
    /**
     * @brief LeakyReLU激活函数
     * @param x 输入值
     * @return 激活后的值
     */
    float leakyReLU(float x);

    /**
     * @brief Softsign激活函数
     * @param x 输入值
     * @return 激活后的值
     */
    float softsign(float x);

private:
    // 网络参数
    QVector<QVector<float>> fc1_weight; // 第一层权重
    QVector<float> fc1_bias;            // 第一层偏置
    QVector<QVector<float>> fc2_weight; // 第二层权重
    QVector<float> fc2_bias;            // 第二层偏置
    QVector<QVector<float>> pi_weight;  // 输出层权重
    QVector<float> pi_bias;             // 输出层偏置

    int inputDims;  // 输入维度 (由加载的模型决定)
    int fc1Dims;    // 第一隐藏层维度 (由加载的模型决定)
    int fc2Dims;    // 第二隐藏层维度 (由加载的模型决定)
    int outputDims; // 输出维度 (通常为2, [ax, ay], 由加载的模型决定)
    int uavId;      // 无人机/智能体 ID

    // 运动控制参数
    QPointF currentVelocity = QPointF(0, 0); // 当前速度 (像素/秒)
    float vMax = 50.0f;                      // 最大速度（像素/秒）
    float timeStep = 0.1f;                   // 时间步长（秒）
};

#endif // MADDPG_AGENT_H