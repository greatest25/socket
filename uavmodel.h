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
 * @brief 无人机推理模型类
 * 
 * 该类用于加载训练好的Actor网络模型参数，并执行前向传播计算，
 * 为无人机提供控制决策。
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
     * @brief 执行推理计算
     * @param observation 观察向量
     * @return 动作向量 [a_x, a_y]
     */
    QPointF inference(const QVector<float> &observation);
    
    /**
     * @brief 获取目标速度矢量
     * @param observation 观察向量
     * @return 速度矢量 [v_x, v_y]
     */
    QPointF getVelocity(const QVector<float> &observation);

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
    
    int inputDims;  // 输入维度
    int fc1Dims;    // 第一隐藏层维度
    int fc2Dims;    // 第二隐藏层维度
    int outputDims; // 输出维度
    
    int uavId;      // 无人机ID
    float vMax = 0.1f; // 最大速度
};

#endif // UAVMODEL_H