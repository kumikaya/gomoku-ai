//! AlphaZero 风格的残差卷积网络
//!
//! 输入：[batch, 4, 15, 15]  四通道棋盘编码
//! 输出：
//!   - 策略分布：[batch, 225] 各合法落子点概率
//!   - 局势价值：[batch, 1]  范围 [-1, 1]（Tanh）
//!
//! 骨干网络包含多个残差块，每个残差块由两个 Conv2d+BatchNorm+ReLU 组成，
//! 最后通过跳跃连接相加。

pub mod residual;
