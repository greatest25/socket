#ifndef UAVMODEL_H
#define UAVMODEL_H

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
 * @brief 无人机推理模型类（适用于3v3场景）
 * 
 * 该类用于加载训练好的Actor网络模型参数，并执行前向传播计算，
 * 为无人机提供控制决策。支持3v3场景的20维观察向量。
 */
class UAVModel : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param id 无人机ID
     * @param parent 父对象指针
     */
    explicit UAVModel(int id = 0, QObject *parent = nullptr);
    
    /**
     * @brief 加载模型参数
     * @param filePath 模型参数JSON文件路径
     * @return 是否加载成功
     */
    bool loadModel(const QString &filePath);
    
    /**
     * @brief 执行推理计算（适用于3v3场景）
     * @param observation 20维观察向量
     * @return 动作向量 [a_x, a_y]
     */
    QPointF inference(const QVector<float> &observation);
    
    /**
     * @brief 获取目标速度矢量（适用于3v3场景）
     * @param observation 20维观察向量
     * @return 速度矢量 [v_x, v_y]
     */
    QPointF getVelocity(const QVector<float> &observation);
    
    /**
     * @brief 构建3v3场景的观察向量
     * @param selfPos 自身位置
     * @param selfVel 自身速度
     * @param friendlyPos 友方无人机位置列表
     * @param enemyPos 敌方无人机位置列表
     * @param targetDistance 目标距离
     * @param targetAngle 目标角度
     * @return 20维观察向量
     */
    QVector<float> buildObservation3v3(const QPointF &selfPos, const QPointF &selfVel,
                                       const QVector<QPointF> &friendlyPos,
                                       const QVector<QPointF> &enemyPos,
                                       float targetDistance, float targetAngle);

    /**
     * @brief 构建观察向量（保持26维格式）
     * @param selfPos 自身位置
     * @param selfVel 自身速度
     * @param friendlyPos 友方无人机位置列表（2个）
     * @param enemyPos 敌方无人机位置列表（3个）
     * @param targetDistance 目标距离
     * @param targetAngle 目标角度
     * @return 26维观察向量
     */
    QVector<float> buildObservation(const QPointF &selfPos, const QPointF &selfVel,
                                   const QVector<QPointF> &friendlyPos,
                                   const QVector<QPointF> &enemyPos,
                                   float targetDistance, float targetAngle);

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
    
    int inputDims;  // 输入维度（3v3场景为20）
    int fc1Dims;    // 第一隐藏层维度
    int fc2Dims;    // 第二隐藏层维度
    int outputDims; // 输出维度
    
    int uavId;      // 无人机ID
    float vMax = 0.1f; // 最大速度
    QPointF currentVelocity = QPointF(0, 0); // 当前速度
    float timeStep = 0.5f;                   // 时间步长
    
    // 3v3场景常量
    static const int OBSERVATION_DIM_3V3 = 20; // 3v3场景观察向量维度
    static const int FRIENDLY_COUNT = 2;       // 友方无人机数量
    static const int ENEMY_COUNT = 3;          // 敌方无人机数量
    static const int LASER_SENSOR_COUNT = 12; // 激光传感器数量
    static const int OBSERVATION_DIM = 26;    // 观察向量维度
    static const int FRIENDLY_COUNT = 2;      // 友方无人机数量
    static const int ENEMY_COUNT = 3;         // 敌方无人机数量
};

#endif // UAVMODEL_H