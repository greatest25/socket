#include "uavmodel.h"
#include <QtMath>

UAVModel::UAVModel(int id, QObject *parent) : QObject(parent),
    inputDims(0), fc1Dims(0), fc2Dims(0), outputDims(0), uavId(id)
{
}

bool UAVModel::loadModel(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "无法打开模型文件:" << filePath;
        return false;
    }
    
    QByteArray jsonData = file.readAll();
    file.close();
    
    QJsonDocument doc = QJsonDocument::fromJson(jsonData);
    if (doc.isNull()) {
        qWarning() << "解析JSON失败:" << filePath;
        return false;
    }
    
    QJsonObject params = doc.object();
    
    // 加载fc1参数
    if (params.contains("fc1.weight")) {
        QJsonArray weightArray = params["fc1.weight"].toArray();
        fc1Dims = weightArray.size();
        if (fc1Dims > 0) {
            inputDims = weightArray[0].toArray().size();
            
            fc1_weight.resize(fc1Dims);
            for (int i = 0; i < fc1Dims; i++) {
                QJsonArray rowArray = weightArray[i].toArray();
                fc1_weight[i].resize(inputDims);
                for (int j = 0; j < inputDims; j++) {
                    fc1_weight[i][j] = rowArray[j].toDouble();
                }
            }
        }
    }
    
    if (params.contains("fc1.bias")) {
        QJsonArray biasArray = params["fc1.bias"].toArray();
        fc1_bias.resize(biasArray.size());
        for (int i = 0; i < biasArray.size(); i++) {
            fc1_bias[i] = biasArray[i].toDouble();
        }
    }
    
    // 加载fc2参数
    if (params.contains("fc2.weight")) {
        QJsonArray weightArray = params["fc2.weight"].toArray();
        fc2Dims = weightArray.size();
        
        fc2_weight.resize(fc2Dims);
        for (int i = 0; i < fc2Dims; i++) {
            QJsonArray rowArray = weightArray[i].toArray();
            fc2_weight[i].resize(fc1Dims);
            for (int j = 0; j < fc1Dims; j++) {
                fc2_weight[i][j] = rowArray[j].toDouble();
            }
        }
    }
    
    if (params.contains("fc2.bias")) {
        QJsonArray biasArray = params["fc2.bias"].toArray();
        fc2_bias.resize(biasArray.size());
        for (int i = 0; i < biasArray.size(); i++) {
            fc2_bias[i] = biasArray[i].toDouble();
        }
    }
    
    // 加载pi参数
    if (params.contains("pi.weight")) {
        QJsonArray weightArray = params["pi.weight"].toArray();
        outputDims = weightArray.size();
        
        pi_weight.resize(outputDims);
        for (int i = 0; i < outputDims; i++) {
            QJsonArray rowArray = weightArray[i].toArray();
            pi_weight[i].resize(fc2Dims);
            for (int j = 0; j < fc2Dims; j++) {
                pi_weight[i][j] = rowArray[j].toDouble();
            }
        }
    }
    
    if (params.contains("pi.bias")) {
        QJsonArray biasArray = params["pi.bias"].toArray();
        pi_bias.resize(biasArray.size());
        for (int i = 0; i < biasArray.size(); i++) {
            pi_bias[i] = biasArray[i].toDouble();
        }
    }
    
    qDebug() << "模型加载成功，输入维度:" << inputDims 
             << "隐藏层1:" << fc1Dims 
             << "隐藏层2:" << fc2Dims 
             << "输出维度:" << outputDims;
    
    return true;
}

QPointF UAVModel::inference(const QVector<float> &observation)
{
    if (observation.size() != inputDims) {
        qWarning() << "输入维度不匹配，期望:" << inputDims << "实际:" << observation.size();
        return QPointF(0, 0);
    }
    
    // 第一层前向传播: x = leaky_relu(fc1(state))
    QVector<float> fc1_output(fc1Dims, 0.0f);
    for (int i = 0; i < fc1Dims; i++) {
        float sum = fc1_bias[i];
        for (int j = 0; j < inputDims; j++) {
            sum += fc1_weight[i][j] * observation[j];
        }
        fc1_output[i] = leakyReLU(sum);
    }
    
    // 第二层前向传播: x = leaky_relu(fc2(x))
    QVector<float> fc2_output(fc2Dims, 0.0f);
    for (int i = 0; i < fc2Dims; i++) {
        float sum = fc2_bias[i];
        for (int j = 0; j < fc1Dims; j++) {
            sum += fc2_weight[i][j] * fc1_output[j];
        }
        fc2_output[i] = leakyReLU(sum);
    }
    
    // 输出层前向传播: pi = softsign(pi(x))
    QVector<float> pi_output(outputDims, 0.0f);
    for (int i = 0; i < outputDims; i++) {
        float sum = pi_bias[i];
        for (int j = 0; j < fc2Dims; j++) {
            sum += pi_weight[i][j] * fc2_output[j];
        }
        pi_output[i] = softsign(sum);
    }
    
    // 限制动作幅度
    float ax = pi_output[0];
    float ay = pi_output[1];
    float magnitude = qSqrt(ax*ax + ay*ay);
    if (magnitude > 0.04f) {
        ax = ax / magnitude * 0.04f;
        ay = ay / magnitude * 0.04f;
    }
    
    return QPointF(ax, ay);
}

QPointF UAVModel::getVelocity(const QVector<float> &observation)
{
    // 执行模型推理得到加速度
    QPointF acceleration = inference(observation);
    
    // 将加速度方向直接映射为速度方向
    float vx = acceleration.x();
    float vy = acceleration.y();
    
    // 限制速度大小
    float magnitude = qSqrt(vx*vx + vy*vy);
    if (magnitude > 0.0f) {
        // 将加速度方向转换为速度方向，并使用最大速度
        vx = vx / magnitude * vMax;
        vy = vy / magnitude * vMax;
    }
    
    return QPointF(vx, vy);
}

float UAVModel::leakyReLU(float x)
{
    return x > 0 ? x : 0.01f * x;
}

float UAVModel::softsign(float x)
{
    return x / (1.0f + qAbs(x));
}
#include <QtMath>

UAVModel::UAVModel(int id, QObject *parent) 
    : QObject(parent)
    , uavId(id)
{
}

QPointF UAVModel::getTargetVelocity(const QVector<float> &observation)
{
    // 执行模型推理得到加速度
    QPointF acceleration = inference(observation); // 这里调用之前实现的inference方法
    
    // 将加速度直接作为速度方向的指示
    float vx = acceleration.x();
    float vy = acceleration.y();
    
    // 限制速度大小
    float magnitude = qSqrt(vx*vx + vy*vy);
    if (magnitude > 0.0f) {
        // 将加速度方向转换为速度方向，并使用最大速度
        vx = vx / magnitude * vMax;
        vy = vy / magnitude * vMax;
    }
    
    return QPointF(vx, vy);
}

QPointF UAVModel::getVelocity(const QVector<float> &observation)
{
    // 执行模型推理得到加速度
    QPointF acceleration = inference(observation); // 这里调用之前实现的inference方法
    
    // 更新速度
    float vx = currentVelocity.x() + acceleration.x() * timeStep;
    float vy = currentVelocity.y() + acceleration.y() * timeStep;
    
    // 限制速度大小
    float magnitude = qSqrt(vx*vx + vy*vy);
    if (magnitude > vMax) {
        vx = vx / magnitude * vMax;
        vy = vy / magnitude * vMax;
    }
    
    currentVelocity = QPointF(vx, vy);
    return currentVelocity;
}