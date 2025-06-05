#include "maddpg_agent.h"
#include <QtMath>

UAVModel::UAVModel(int id, QObject *parent) : QObject(parent),
    inputDims(0), fc1Dims(0), fc2Dims(0), outputDims(0), uavId(id) // uavId 初始化
{
}

// 加载 Actor 模型参数 (从JSON文件)
bool UAVModel::loadModel(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "UAVModel (ID:" << uavId << ") - 无法打开模型文件:" << filePath;
        return false;
    }

    QByteArray jsonData = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(jsonData);
    if (doc.isNull()) {
        qWarning() << "UAVModel (ID:" << uavId << ") - 解析JSON失败:" << filePath;
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
            fc2_weight[i].resize(fc1Dims); // fc2的输入是fc1的输出维度
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

    // 加载pi参数 (输出层)
    if (params.contains("pi.weight")) {
        QJsonArray weightArray = params["pi.weight"].toArray();
        outputDims = weightArray.size();

        pi_weight.resize(outputDims);
        for (int i = 0; i < outputDims; i++) {
            QJsonArray rowArray = weightArray[i].toArray();
            pi_weight[i].resize(fc2Dims); // pi的输入是fc2的输出维度
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

    qDebug() << "UAVModel (ID:" << uavId << ") - 模型加载成功，输入维度:" << inputDims
             << "隐藏层1:" << fc1Dims
             << "隐藏层2:" << fc2Dims
             << "输出维度:" << outputDims;

    return true;
}

// 执行 Actor 网络前向传播，获取归一化动作
QPointF UAVModel::inference(const QVector<float> &observation)
{
    if (inputDims == 0) { // 检查模型是否已成功加载并初始化维度
        qWarning() << "UAVModel (ID:" << uavId << ") - 模型未加载或维度未初始化，无法执行推理。";
        return QPointF(0, 0);
    }
    if (observation.size() != inputDims) {
        qWarning() << "UAVModel (ID:" << uavId << ") - 输入维度不匹配，期望:" << inputDims << "实际:" << observation.size();
        return QPointF(0, 0);
    }

    // 第一层前向传播: x = leaky_relu(fc1(state))
    QVector<float> fc1_output(fc1Dims, 0.0f);
    for (int i = 0; i < fc1Dims; i++) {
        float sum = fc1_bias.size() > i ? fc1_bias[i] : 0.0f; // 安全访问
        for (int j = 0; j < inputDims; j++) {
            if (fc1_weight.size() > i && fc1_weight[i].size() > j) { // 安全访问
                 sum += fc1_weight[i][j] * observation[j];
            } else {
                qWarning() << "UAVModel (ID:" << uavId << ") - 访问越界: fc1_weight[" << i << "][" << j << "]";
                return QPointF(0,0); // 提前返回错误
            }
        }
        fc1_output[i] = leakyReLU(sum);
    }

    // 第二层前向传播: x = leaky_relu(fc2(x))
    QVector<float> fc2_output(fc2Dims, 0.0f);
    for (int i = 0; i < fc2Dims; i++) {
        float sum = fc2_bias.size() > i ? fc2_bias[i] : 0.0f; // 安全访问
        for (int j = 0; j < fc1Dims; j++) {
             if (fc2_weight.size() > i && fc2_weight[i].size() > j) { // 安全访问
                sum += fc2_weight[i][j] * fc1_output[j];
            } else {
                qWarning() << "UAVModel (ID:" << uavId << ") - 访问越界: fc2_weight[" << i << "][" << j << "]";
                return QPointF(0,0); // 提前返回错误
            }
        }
        fc2_output[i] = leakyReLU(sum);
    }

    // 输出层前向传播: pi = softsign(pi(x))
    QVector<float> pi_output(outputDims, 0.0f);
     if (outputDims > 0) { // 确保输出层已定义
        for (int i = 0; i < outputDims; i++) {
            float sum = pi_bias.size() > i ? pi_bias[i] : 0.0f; // 安全访问
            for (int j = 0; j < fc2Dims; j++) {
                if (pi_weight.size() > i && pi_weight[i].size() > j) { // 安全访问
                    sum += pi_weight[i][j] * fc2_output[j];
                } else {
                    qWarning() << "UAVModel (ID:" << uavId << ") - 访问越界: pi_weight[" << i << "][" << j << "]";
                    return QPointF(0,0); // 提前返回错误
                }
            }
            pi_output[i] = softsign(sum);
        }
    } else {
        qWarning() << "UAVModel (ID:" << uavId << ") - 输出层维度为0，无法计算动作。";
        return QPointF(0,0);
    }


    // softsign 的输出 ax 和 ay 已经在 (-1, 1) 区间内
    // 无需额外缩放即可满足 [-1, 1] 的归一化动作要求
    float ax = (outputDims > 0 && pi_output.size() > 0) ? pi_output[0] : 0.0f;
    float ay = (outputDims > 1 && pi_output.size() > 1) ? pi_output[1] : 0.0f;

    return QPointF(ax, ay);
}

QPointF UAVModel::getTargetVelocity(const QVector<float> &observation)
{
    // 执行模型推理得到加速度方向分量
    QPointF acceleration_components = inference(observation);

    // 将加速度方向直接映射为速度方向
    float vx_dir = acceleration_components.x();
    float vy_dir = acceleration_components.y();

    float magnitude = qSqrt(vx_dir*vx_dir + vy_dir*vy_dir);
    if (magnitude > 1e-6f) { // 避免除以零或非常小的值
        // 将方向向量归一化，然后乘以最大速度
        return QPointF(vx_dir / magnitude * vMax, vy_dir / magnitude * vMax);
    }
    return QPointF(0, 0); // 如果输出接近零向量，则目标速度为零
}

QPointF UAVModel::getVelocity(const QVector<float> &observation)
{
    // 执行模型推理得到归一化的加速度分量
    QPointF normalized_accel = inference(observation);

    // 实际加速度值 (这里假设 normalized_accel 的每个分量直接代表了加速度的一个比例因子，
    // 乘以一个合适的标度。如果动作空间直接输出速度，则此逻辑不同)
    // 假设 timeStep 内加速度恒定，可以简单地认为是 timeStep 内速度的变化量与最大速度的比例。
    // 或者，可以定义一个 uav_accel_scale, 如Python环境中的 uav_input_accel_scale。
    // 为简单起见，这里假设 inference() 的输出直接按比例影响速度变化，并受到 timeStep 影响。
    // 这个转换逻辑需要与Python环境中的动作解释方式严格对应。
    // 此处的示例将归一化动作视为加速度与最大加速度的比例。
    // float accel_scale_factor = 10.0f; // 假设一个加速度标度，需要调整

    // 更新速度: v_new = v_current + a * dt
    // 这里的 a 是实际的加速度。如果 inference() 输出的是一个与最大加速度的比例，
    // 则 a = normalized_accel * max_physical_acceleration。
    // 为了简化，我们暂时不引入 max_physical_acceleration，而是让动作直接影响速度变化，
    // 并通过 vMax 进行裁剪。这种解释方式可能需要根据实际物理模型调整。

    float vx = currentVelocity.x() + normalized_accel.x() * vMax * timeStep; // 简化的模型，normalized_accel 影响vMax的一部分
    float vy = currentVelocity.y() + normalized_accel.y() * vMax * timeStep; // 简化的模型

    // 限制速度大小
    float magnitude = qSqrt(vx*vx + vy*vy);
    if (magnitude > vMax) {
        vx = vx / magnitude * vMax;
        vy = vy / magnitude * vMax;
    }

    currentVelocity = QPointF(vx, vy);
    return currentVelocity;
}

float UAVModel::leakyReLU(float x)
{
    return x > 0 ? x : 0.01f * x;
}

float UAVModel::softsign(float x)
{
    return x / (1.0f + qAbs(x));
}