//! 训练指标记录
//!
//! 基于 burn 的 `FileMetricLogger`，将每个 batch 的指标写入结构化日志文件。
//! 日志目录结构：
//! ```text
//! {log_dir}/train/epoch-{N}/Total_Loss.log
//! {log_dir}/train/epoch-{N}/Policy_Loss.log
//! {log_dir}/train/epoch-{N}/Value_Loss.log
//! {log_dir}/train/epoch-{N}/Entropy.log
//! {log_dir}/train/epoch-{N}/Explained_Variance.log
//! ```
//!
//! 训练结束后可以用 `burn::train::LearnerSummary` 读取并打印 Min/Max 汇总表。

use burn::train::logger::{FileMetricLogger, MetricLogger};
use burn::train::metric::store::{MetricsUpdate, NumericMetricUpdate, Split};
use burn::train::metric::{
    MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAggregation,
    NumericAttributes, NumericEntry, SerializedEntry,
};
use std::path::Path;
use std::sync::Arc;

// ── 指标 ID ──

const METRIC_TOTAL_LOSS: &str = "Total Loss";
const METRIC_POLICY_LOSS: &str = "Policy Loss";
const METRIC_VALUE_LOSS: &str = "Value Loss";
const METRIC_ENTROPY: &str = "Entropy";
const METRIC_EXPLAINED_VAR: &str = "Explained Variance";

/// 封装 burn `FileMetricLogger`，提供简洁的标量记录接口。
pub struct TrainingLogger {
    logger: FileMetricLogger,
    /// 是否已注册 metric definitions（首次 log 时自动注册）
    registered: bool,

    total_loss_id: MetricId,
    policy_loss_id: MetricId,
    value_loss_id: MetricId,
    entropy_id: MetricId,
    explained_var_id: MetricId,
}

impl TrainingLogger {
    /// 创建 logger，日志写入 `log_dir` 目录下。
    pub fn new(log_dir: &Path) -> Self {
        Self {
            logger: FileMetricLogger::new(log_dir),
            registered: false,
            total_loss_id: metric_id(METRIC_TOTAL_LOSS),
            policy_loss_id: metric_id(METRIC_POLICY_LOSS),
            value_loss_id: metric_id(METRIC_VALUE_LOSS),
            entropy_id: metric_id(METRIC_ENTROPY),
            explained_var_id: metric_id(METRIC_EXPLAINED_VAR),
        }
    }

    /// 首次调用时向 logger 注册所有指标的定义信息。
    fn ensure_registered(&mut self) {
        if self.registered {
            return;
        }
        let defs = vec![
            make_definition(
                self.total_loss_id.clone(),
                METRIC_TOTAL_LOSS,
                false, /* lower is better */
            ),
            make_definition(self.policy_loss_id.clone(), METRIC_POLICY_LOSS, false),
            make_definition(self.value_loss_id.clone(), METRIC_VALUE_LOSS, false),
            make_definition(self.entropy_id.clone(), METRIC_ENTROPY, false),
            make_definition(
                self.explained_var_id.clone(),
                METRIC_EXPLAINED_VAR,
                true, /* higher is better */
            ),
        ];
        for def in defs {
            self.logger.log_metric_definition(def);
        }
        self.registered = true;
    }

    /// 记录单个 batch 的训练指标。
    ///
    /// 每调用一次即在对应 epoch 的 `.log` 文件中追加一行。
    /// `FileMetricLogger` 在 epoch 变化时自动切换到新文件。
    pub fn log_batch(
        &mut self,
        epoch: usize,
        total_loss: f32,
        policy_loss: f32,
        value_loss: f32,
        entropy: f32,
    ) {
        self.ensure_registered();

        let updates = vec![
            make_numeric_update(self.total_loss_id.clone(), total_loss),
            make_numeric_update(self.policy_loss_id.clone(), policy_loss),
            make_numeric_update(self.value_loss_id.clone(), value_loss),
            make_numeric_update(self.entropy_id.clone(), entropy),
        ];

        self.logger
            .log(MetricsUpdate::new(vec![], updates), epoch, &Split::Train);
    }

    /// 记录 epoch 级别的汇总指标（如 explained_variance）。
    ///
    /// 每个 epoch 仅调用一次。
    pub fn log_epoch_summary(&mut self, epoch: usize, explained_variance: f32) {
        self.ensure_registered();

        let update = make_numeric_update(self.explained_var_id.clone(), explained_variance);
        self.logger.log(
            MetricsUpdate::new(vec![], vec![update]),
            epoch,
            &Split::Train,
        );
    }
}

// ── 辅助函数 ──

fn metric_id(name: &str) -> MetricId {
    MetricId::new(Arc::new(name.to_string()))
}

fn make_definition(id: MetricId, name: &str, higher_is_better: bool) -> MetricDefinition {
    MetricDefinition {
        metric_id: id,
        name: name.to_string(),
        description: None,
        attributes: MetricAttributes::Numeric(NumericAttributes {
            unit: None,
            higher_is_better,
            aggregation: NumericAggregation::Mean,
        }),
    }
}

/// 构造一个数值指标更新条目。
///
/// `serialized` 格式为 `NumericEntry::Value(value)`，
/// 文件内容为每行一个 f64 字符串，方便后续用 `LearnerSummary` 聚合。
fn make_numeric_update(id: MetricId, value: f32) -> NumericMetricUpdate {
    let num_entry = NumericEntry::Value(value as f64);
    let serialized = num_entry.serialize(); // 即 value.to_string()
    let formatted = format!("{:.4}", value);
    let entry = MetricEntry::new(id, SerializedEntry::new(formatted, serialized));
    NumericMetricUpdate::new(entry, num_entry.clone(), num_entry)
}

// ── 批次统计（用于控制台输出 + logger 记录） ──

/// 单批次训练产生的指标快照。
#[derive(Clone, Debug)]
pub struct BatchStats {
    pub total_loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub batch_size: usize,
}

/// 一个 epoch 训练阶段的累计统计。
#[derive(Clone, Debug, Default)]
pub struct EpochStats {
    pub total_loss_sum: f32,
    pub policy_loss_sum: f32,
    pub value_loss_sum: f32,
    pub entropy_sum: f32,
    pub num_batches: usize,
    pub total_samples: usize,
    /// 收集所有 batch 的价值预测，用于计算 explained_variance
    pub all_value_preds: Vec<f32>,
    /// 收集所有 batch 的价值目标，用于计算 explained_variance
    pub all_value_targets: Vec<f32>,
}

impl EpochStats {
    pub fn push(&mut self, batch: BatchStats) {
        self.total_loss_sum += batch.total_loss;
        self.policy_loss_sum += batch.policy_loss;
        self.value_loss_sum += batch.value_loss;
        self.entropy_sum += batch.entropy;
        self.num_batches += 1;
        self.total_samples += batch.batch_size;
    }

    pub fn avg_total_loss(&self) -> f32 {
        if self.num_batches == 0 {
            0.0
        } else {
            self.total_loss_sum / self.num_batches as f32
        }
    }
    pub fn avg_policy_loss(&self) -> f32 {
        if self.num_batches == 0 {
            0.0
        } else {
            self.policy_loss_sum / self.num_batches as f32
        }
    }
    pub fn avg_value_loss(&self) -> f32 {
        if self.num_batches == 0 {
            0.0
        } else {
            self.value_loss_sum / self.num_batches as f32
        }
    }
    pub fn avg_entropy(&self) -> f32 {
        if self.num_batches == 0 {
            0.0
        } else {
            self.entropy_sum / self.num_batches as f32
        }
    }
}
