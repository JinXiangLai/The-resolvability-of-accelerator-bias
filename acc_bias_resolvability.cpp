#include <chrono>
#include <iostream>
#include <map>
#include <fstream>
#include <random>
#include <vector>
#include <Eigen/Dense>

using namespace std;
// 观察acc bias的可观性，研究acc bias如何激励出来
// 输入：不含尺度的pose、gyroscope读数、accelerator读数、相应噪声、重力(W系下)、关键帧个数、采样时间(图像、IMU)、gyro & acc bias
// 解算：重力(c0系下)、尺度、acc bias、恢复各时刻速度

/*===========================================================*/
/*================ Control Parameter ========================*/
/*===========================================================*/
constexpr int kImuFrequency = 120; // hz
constexpr double kImuSamplePeriod = 1.0 / 120; // s
constexpr int kKeyFrameFrequency = 1; // hz
const int kImu2ImgRate = kImuFrequency / kKeyFrameFrequency; // 每隔多少个IMU取一个关键帧
constexpr double kScale = 0.5;
constexpr double kRad2Deg = 180 / M_PI;
constexpr double kDeg2Rad = M_PI / 180;
constexpr double kGravityValue = 9.80;
constexpr bool kAddNoise2Acc = true;
constexpr double kAccNoiseStd = 0.1; // noise相比acc真实值过大时，将影响acc bias的可观性
constexpr bool kAddNoise2Gyr = true;
constexpr double kGyrNoiseStd = 0.01 * kDeg2Rad; // gyro noise会极大地影响 acc bias的可观性
const Eigen::Vector3d v0(0.1, 0.2, 0.1); // IMU 需要考虑初始速度, c0系下
const Eigen::Vector3d accBias(0.2, 0.19, 0.05);
const Eigen::Vector3d gyrBias(0.0, 0.0, 0.0);
Eigen::Vector3d G_c0(-1, -2, -3); // c0 系下的实际重力(含方向)，认为c系与b系重合

/*===========================================================*/
/*=================== Declaration ===========================*/
/*===========================================================*/
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

void GenerateIMUdata(const int keyframeNum, vector<Eigen::Vector3d> &gyr, vector<Eigen::Vector3d> &acc);

void GeneratePreintergrateQPVandQPVandPreintergrateJacobian(const vector<Eigen::Vector3d> &gyr, const vector<Eigen::Vector3d> &acc, const int keyframeNum,
        vector<Eigen::Quaterniond> &dQ, vector<Eigen::Vector3d> &dP, vector<Eigen::Vector3d> &dV, 
        vector<Eigen::Quaterniond> &Q, vector<Eigen::Vector3d> &P, vector<Eigen::Vector3d> &V, 
        vector<Eigen::Matrix3d> &Jdp, vector<Eigen::Matrix3d> &Jdv);

void ScaleP(vector<Eigen::Vector3d> &P);

Eigen::Vector4d EvaluateScaleAndGc0(const vector<Eigen::Quaterniond> &dQ, const vector<Eigen::Vector3d> &dP, const vector<Eigen::Vector3d> &dV, 
        const vector<Eigen::Quaterniond> &Q, const vector<Eigen::Vector3d> &P, const vector<Eigen::Vector3d> &V);

void UnScaleP(const double scale, vector<Eigen::Vector3d> &P);

Eigen::Matrix3d CalculateRc0_w(const Eigen::Vector3d &g_c0);

Eigen::Matrix<double, 6, 1> UpdateScaleAndRc0_wAndAccBias(const Eigen::Matrix3d &Rc0_w, const vector<Eigen::Quaterniond> &Q, const vector<Eigen::Vector3d> &P,
    const vector<Eigen::Vector3d> &dP, const vector<Eigen::Vector3d> &dV,
    const vector<Eigen::Matrix3d> &Jdp, const vector<Eigen::Matrix3d> &Jdv);

void UpdateRc0_w(const double theta_x, const double theta_y, Eigen::Matrix3d &Rc0_w);

vector<Eigen::Vector3d> RecoverVelocity(const vector<Eigen::Quaterniond> &Q, const vector<Eigen::Vector3d> &P, const vector<Eigen::Vector3d> &dP, 
    const vector<Eigen::Vector3d> &dV, const vector<Eigen::Matrix3d> &Jdp, const vector<Eigen::Matrix3d> &Jdv, 
    const Eigen::Vector3d accBias, const Eigen::Matrix3d &Rc0_w);

/*========================================================*/
/*=================== Pipeline ===========================*/
/*========================================================*/
int main(){

    const int keyframeNum = 6;
    vector<Eigen::Vector3d> gyr, acc;
    GenerateIMUdata(keyframeNum, gyr, acc);
    vector<Eigen::Quaterniond> dQ, Q;
    vector<Eigen::Vector3d> dP, dV, P, V;
    vector<Eigen::Matrix3d> Jdp, Jdv; // dP, dV关于accBias的导数
    GeneratePreintergrateQPVandQPVandPreintergrateJacobian(gyr, acc, keyframeNum, dQ, dP, dV, Q, P, V, Jdp, Jdv);
    
    ScaleP(P);
    Eigen::Vector4d scaleAndGc0 = EvaluateScaleAndGc0(dQ, dP, dV, Q, P, V);
    UnScaleP(scaleAndGc0[0], P);
    
    Eigen::Matrix3d Rc0_w = CalculateRc0_w(scaleAndGc0.tail(3));

    const Eigen::Matrix<double, 6, 1> s_theta_ba = UpdateScaleAndRc0_wAndAccBias(Rc0_w, Q, P, dP, dV, Jdp, Jdv);
    UnScaleP(s_theta_ba[0], P);
    // 求解出水平姿态角度增量后，必须对Rc0_w进行更新，失之毫厘，差之千里，这就是数学的严谨
    UpdateRc0_w(s_theta_ba[1], s_theta_ba[2], Rc0_w);

    const vector<Eigen::Vector3d> v = RecoverVelocity(Q, P, dP, dV, Jdp, Jdv, s_theta_ba.tail(3), Rc0_w);
    for(int i = 0; i < Q.size(); ++i) {
        cout << "diff v[" << i << "] - V[" << i << "]" << (v[i] - V[i]).norm() << endl;
    }

    return 0;
}

/*=======================================================*/
/*================= Realization =========================*/
/*=======================================================*/
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m;
    m.setZero();
    m << 0, -v[2], v[1],
         v[2], 0, -v[0],
         -v[1], v[0], 0;
    return m;
}

void GenerateIMUdata(const int keyframeNum, vector<Eigen::Vector3d> &gyr, vector<Eigen::Vector3d> &acc) {
    // 任意相邻2个IMU读数可以构成一个dt间隔，所需的IMU数量即：(N-1) * imuRate + 1
    const int imuDataNum = (keyframeNum - 1) * kImu2ImgRate + 1;
    Eigen::Vector3d g_c0_inv = -1.0 * G_c0.normalized() * kGravityValue;
    // g_c0系下实际的重力为 -gc0
    std::default_random_engine generator;
    // 均匀分布数值
    std::uniform_real_distribution<double> accReader(-0.01, 0.01);
    std::uniform_real_distribution<double> gyrReader(-0.01 * kDeg2Rad, 0.01 * kDeg2Rad);

    for(int i = 0; i < imuDataNum; ++i) {
        // 非退化场景
        Eigen::Vector3d a(accReader(generator), accReader(generator), accReader(generator));
        // 退化场景
        // Eigen::Vector3d a(0, 0, 0);
        // acc 读数需考虑重力的反向矢量在该系下的测量值
        a += g_c0_inv;
        acc.push_back(a);
        
        // 非退化场景
        // Eigen::Vector3d w(gyrReader(generator), gyrReader(generator), gyrReader(generator));
        // 退化场景
        Eigen::Vector3d w(0, 0, gyrReader(generator));
        gyr.push_back(w);
    }
}

void GeneratePreintergrateQPVandQPVandPreintergrateJacobian(const vector<Eigen::Vector3d> &gyr, const vector<Eigen::Vector3d> &acc, const int keyframeNum,
        vector<Eigen::Quaterniond> &dQ, vector<Eigen::Vector3d> &dP, vector<Eigen::Vector3d> &dV, 
        vector<Eigen::Quaterniond> &Q, vector<Eigen::Vector3d> &P, vector<Eigen::Vector3d> &V,
        vector<Eigen::Matrix3d> &Jdp, vector<Eigen::Matrix3d> &Jdv) {
    const int imgDurationNum = (keyframeNum - 1);
    // 预积分量
    dQ.resize(keyframeNum - 1);
    dP.resize(keyframeNum - 1);
    dV.resize(keyframeNum - 1);
    Jdp.resize(keyframeNum - 1);
    Jdv.resize(keyframeNum - 1);
    // 世界系下的量
    Q.resize(keyframeNum);
    P.resize(keyframeNum);
    V.resize(keyframeNum);
    Q[0] = Eigen::Quaterniond::Identity();
    P[0] = Eigen::Vector3d::Zero();
    V[0] = v0;
    Eigen::Vector3d g_c0 = G_c0.normalized() * kGravityValue;
    const double dt = kImuSamplePeriod;
    const double dt2 = std::pow(dt, 2);

    int id = 0; // 索引当前预积分开始的时段
    for(int i = 0; i < imgDurationNum; ++i) {
        // endId处认为触发图像
        const int endId = id + kImu2ImgRate;
        dQ[i].setIdentity();
        dP[i].setZero();
        dV[i].setZero();

        Jdp[i].setZero();
        Jdv[i].setZero();

        Q[i+1].setIdentity();
        P[i+1].setZero();
        V[i+1].setZero();

        // 不含bias的预积分
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        Eigen::Vector3d v = Eigen::Vector3d::Zero();

        // Lambda expression
        auto DeltaQ = [](const Eigen::Vector3d& angle) -> Eigen::Quaterniond{
            Eigen::Matrix3d Rc1c2 = Eigen::AngleAxisd(angle.norm(), angle.normalized()).toRotationMatrix();
            Eigen::Quaterniond DeltaQ(Rc1c2);
            DeltaQ.normalize();
            return DeltaQ;
        };

        for(; id < endId; ++id){
            // 使用中值积分
            const Eigen::Vector3d w = (gyr[id] + gyr[id+1]) * 0.5;
            const Eigen::Vector3d a = (acc[id] + acc[id+1]) * 0.5;
            // bias扰动的IMU读数
            Eigen::Vector3d wb = w + gyrBias;
            Eigen::Vector3d ab = a + accBias;
            // 高斯分布噪声
            std::default_random_engine generator;
            std::normal_distribution<double> accNoise(0, kAccNoiseStd);
            std::normal_distribution<double> gyrNoise(0, kGyrNoiseStd);
            if(kAddNoise2Acc) {
                const Eigen::Vector3d noise(accNoise(generator), accNoise(generator), accNoise(generator));
                // noise扰动acc读数
                ab += noise;
            }
            if(kAddNoise2Gyr) {
                const Eigen::Vector3d noise(gyrNoise(generator), gyrNoise(generator), gyrNoise(generator));
                // noise扰动gyr读数
                wb += noise;
            }

            // 含bias的预积分
            dP[i] += dV[i] * dt + 0.5 * (dQ[i] * ab) * dt2;
            dV[i] += dQ[i] * ab * dt;
            // 关于bias的雅可比
            Jdp[i] += Jdv[i] * dt - 0.5 * dQ[i].toRotationMatrix() * dt2;
            Jdv[i] -= dQ[i].toRotationMatrix() * dt;
            // 最后再更新旋转预积分
            dQ[i] *=  DeltaQ(wb * dt);

            // 计算扣除bias后的真实值
            p += v * dt + 0.5 * (q * a) * dt2;
            v += q * a * dt;
            q *= DeltaQ(w * dt);
        }
        
        const double dT = dt * kImu2ImgRate;
        // 不考虑accBias的c0系下状态量表示
        P[i+1] = P[i] + V[i] * dT + Q[i] * p + 0.5 * g_c0  * std::pow(dT, 2);
        V[i+1] = V[i] + Q[i] * v + g_c0 * dT;
        Q[i+1] = Q[i] * q;
    }
}

void ScaleP(vector<Eigen::Vector3d> &P) {
    const int size = P.size();
    // SFM 能够提供准确的不包含尺度的poses
    for(int i = 0; i < size; ++i) {
        P[i] *= kScale;
        // V[i] *= kScale; // V是需要求解的
    }
}

void UnScaleP(const double scale, vector<Eigen::Vector3d> &P) {
    const int size = P.size();
    for(int i = 0; i < size; ++i) {
        P[i] *= scale;
        // V[i] *= scale;
    }
}

// 首先求解scale 和 Gc0，这里忽略相机和IMU之间的外参，
// 同时认为任意相邻关键帧间的时间间隔dT相等
Eigen::Vector4d EvaluateScaleAndGc0(const vector<Eigen::Quaterniond> &dQ, const vector<Eigen::Vector3d> &dP, const vector<Eigen::Vector3d> &dV, 
        const vector<Eigen::Quaterniond> &Q, const vector<Eigen::Vector3d> &P, const vector<Eigen::Vector3d> &V) {
    /******** 数学原理 ********
    * sP1 = sP0 + sV0 * dT + 0.5 * Gc0 * dT^2 + Rc0_b0 * dP0 (1)
    * sP2 = sP1 + sV1 * dT + 0.5 * Gc0 * dT^2 + Rc0_b1 * dP1 (2)
    * 这里一个简单的地方是，我们假设关键帧采样是均匀的，所以Gc0仅在速度项存在
    * (2) - (1)得:
    * s(P2 - P1) = s(P1 - P0) + [Gc0 * dT + Rc0_b0 * dV0] + Rc0_b1 * dP1 - Rc0_b0 * dP0
    * 移项，合并同类项得:
    * [(2P1 - P0 - P2) * s] + [I * dT * Gc0] + Rc0_b0 * dV0 = Rc0_b0 * dP0 - Rc0_b1 * dP1
    * 将上式写成矩阵形式，A[3x4] * [s, Gc0]' = Rc0_b0 * dP0 - Rc0_b1 * dP1 - Rc0_b0 * dV0
    * 每相邻3个Keyframe可以提供3个等式，因此N个Keyframe可以提供(N-2)*3个等式，要解4个未知数，至少需要
    * Keyframe数量N >= 4
    ********/

    const int KFnum = Q.size(); // 首帧为世界系
    const int row = (KFnum - 2) * 3;
    Eigen::MatrixXd A(row, 4);
    Eigen::VectorXd b(row);
    const double dT = kImuSamplePeriod * kImu2ImgRate; // 理论上，时间间隔应该是IMU个数-1？

    int rowId = 0;
    for(int i = 1; i < Q.size() - 1; ++i) {
        const Eigen::Vector3d p = 2 * P[i] - P[i-1] - P[i+1];
        const Eigen::Matrix3d dtMatrix = dT *Eigen::Matrix3d::Identity();
        const Eigen::Vector3d bi = Q[i-1] * dP[i-1] - Q[i] * dP[i] - Q[i-1] * dV[i-1];
        A.block(rowId, 0, 3, 1) = p;
        A.block(rowId, 1, 3, 3) = dtMatrix;
        b.middleRows(rowId, 3) = bi;
        rowId += 3;
    }

    Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);
    cout << "scale | Gc0: " << x[0] << " " <<x.tail(3).transpose() << endl;
    cout << "Gc0.norm: " << x.tail(3).norm() << endl;
    return x;
}

Eigen::Matrix3d CalculateRc0_w(const Eigen::Vector3d &g_c0) {
    // g_c0 = Rc0_w * g_w
    // 单矢量定姿法求解水平姿态角，yaw角置0
    const Eigen::Vector3d g_w_norm(0, 0, -1);
    const Eigen::Vector3d g_c0_norm = g_c0.normalized();
    // 轴角解算
    const Eigen::Vector3d angleAxis = g_w_norm.cross(g_c0_norm);
    Eigen::Matrix3d Rc0_w = Eigen::AngleAxisd(angleAxis.norm(), angleAxis.normalized()).toRotationMatrix();
    cout << "g_c0_norm: " << g_c0_norm.transpose() << endl;
    cout << "Rc0_w * g_w_norm: " << (Rc0_w * g_w_norm).transpose() << endl;
    // TODO: 设置yaw角值为0
    return Rc0_w;
}

// Gc0系下，求解accBias等
Eigen::Matrix<double, 6, 1> UpdateScaleAndRc0_wAndAccBias(const Eigen::Matrix3d &Rc0_w, const vector<Eigen::Quaterniond> &Q, const vector<Eigen::Vector3d> &P,
    const vector<Eigen::Vector3d> &dP, const vector<Eigen::Vector3d> &dV,
    const vector<Eigen::Matrix3d> &Jdp, const vector<Eigen::Matrix3d> &Jdv){
    /******** 数学原理 ********
    * [(2P1 - P0 - P2) * s] + [I * dT * Gc0] + Rc0_b0 * dV0 = Rc0_b0 * dP0 - Rc0_b1 * dP1
    * 代入水平姿态旋转矩阵及已知Gw，得:
    * [(2P1 - P0 - P2) * s] + [dT * Rc0_w * Gw] = Rc0_b0 * dP0 - Rc0_b1 * dP1 - Rc0_b0 * dV0
    * 难点在于，如何从右边的预积分中分离出accBias？
    * 这里只能采用对预积分进行一阶泰勒关于AccBias的展开，参考:《On-Manifold Preintegration for Real-Time》.
    * 使用李代数扰动Rc0_w，由于yaw角不可观，因此只关注θx, θy，同时代人预积分关于AccBias的一阶泰勒展开式
    * [(2P1 - P0 - P2) * s] + [dT * Rc0_w * exp(ε) * Gw] = Rc0_b0 * (dP0 + Jp0 * ba) - Rc0_b1 * (dP1 + Jp1 * ba) - Rc0_b0 * (dV0 + Jv0 * ba)
    * 将待优化量: s, θx, θy, ba移到左边，共有6个变量待优化，任意相邻3个KF可以提供2个等式，于是(N-2) * 2 >= 6, 即N >=5
    * [(2P1 - P0 - P2) * s] + [dT * Rc0_w * exp(ε) * Gw] - (Rc0_b0 * Jp0 - Rc0_b1 * Jp1 - Rc0_b0 * Jv0) * ba = 
    * Rc0_b0 * dP0 - Rc0_b1 * dP1 - Rc0_b0 * dV0
    * 其中: 
    * dT * Rc0_w * exp(ε) * Gw = dT * Rc0_w *  Gw - dT * Rc0_w * Gw^ * ε
    *************************/
    const int KFnum = Q.size();
    Eigen::MatrixXd A((KFnum - 2) * 3, 6);
    Eigen::VectorXd b((KFnum - 2) * 3);
    const double dt = kImuSamplePeriod;
    const double dT = dt * kImu2ImgRate;
    const Eigen::Vector3d Gw(0, 0, -kGravityValue);

    // 填充A, b矩阵
    int rowId = 0;
    for(int i = 1; i < Q.size()-1; ++i) {
        const Eigen::Vector3d pi = 2 * P[i] - P[i-1] - P[i+1];
        const Eigen::Matrix3d mg = -dT * Rc0_w * skewSymmetric(Gw);
        const Eigen::Matrix3d ma = Q[i-1].toRotationMatrix() * Jdp[i-1] - Q[i].toRotationMatrix() * Jdp[i] - Q[i-1].toRotationMatrix() * Jdv[i-1];
        const Eigen::Vector3d bi = Q[i-1].toRotationMatrix() * dP[i-1] - Q[i] * dP[i] - Q[i-1].toRotationMatrix() * dV[i-1] - dT * Rc0_w * Gw;
        A.block(rowId, 0, 3, 1) = pi;
        // 我们只关心θx, θy的增量，所以mg只取前2列
        A.block(rowId, 1, 3, 2) = mg.block(0, 0, 3, 2);
        A.block(rowId, 3, 3, 3) = -ma;
        b.middleRows(rowId, 3) = bi;
        rowId += 3;
    }
    // cout << "A:\n" << A << endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    // 查看矩阵的秩
    cout << "A's singularValues: " << svd.singularValues().transpose() << endl;

    const Eigen::Matrix<double, 6, 1> s_theta_ba = A.colPivHouseholderQr().solve(b);
    cout << "scale: " << s_theta_ba[0] << endl;
    cout << "theta: " << s_theta_ba.middleRows(1, 2).transpose() * kRad2Deg << endl;
    cout << "ba: " << s_theta_ba.tail(3).transpose() << endl;
    return s_theta_ba;
}

void UpdateRc0_w(const double theta_x, const double theta_y, Eigen::Matrix3d &Rc0_w){
    Eigen::Vector3d theta(theta_x, theta_y, 0);
    // 更新重力方向的旋转量
    Rc0_w = Rc0_w * Eigen::AngleAxisd(theta.norm(), theta.normalized());
}

vector<Eigen::Vector3d> RecoverVelocity(const vector<Eigen::Quaterniond> &Q, const vector<Eigen::Vector3d> &P, const vector<Eigen::Vector3d> &dP, 
    const vector<Eigen::Vector3d> &dV, const vector<Eigen::Matrix3d> &Jdp, const vector<Eigen::Matrix3d> &Jdv, 
    const Eigen::Vector3d accBias, const Eigen::Matrix3d &Rc0_w) {
    /******** 数学原理 ********
    * sP1 = sP0 + sV0 * dT + 0.5 * Gc0 * dT^2 + Rc0_b0 * dP0 (1)
    */
    const int KFnum = Q.size();
    const Eigen::Vector3d Gc0 = Rc0_w * Eigen::Vector3d(0, 0, -kGravityValue);
    const double dT = kImu2ImgRate * kImuSamplePeriod;
    const double dT2 = dT * dT;
    vector<Eigen::Vector3d> v(KFnum);
    // 求解方法1：
    // v[0] = (P[1] - P[0] - 0.5 * Gc0 * dT2 - Q[0] * (dP[0] + Jdp[0] * accBias)) / dT;
    // for(int i = 1; i < v.size(); ++i) {
    //     v[i] = v[i-1] + Gc0 * dT + Q[i-1] * (dV[i-1] + Jdv[i-1] * accBias); 
    // }
    // 求解方法2：
    for(int i = 0; i < v.size()-1; ++i) {
        v[i] = (P[i+1] - P[i] - 0.5 * Gc0 * dT2 - Q[i] * (dP[i] + Jdp[i] * accBias)) / dT;
    }
    v[KFnum-1] = v[KFnum-2] + Gc0 * dT + Q[KFnum-2] * (dV[KFnum-2] + Jdv[KFnum-2] * accBias);
    return v;
}
