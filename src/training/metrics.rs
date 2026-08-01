//! 训练指标记录
//!
//! 基于 burn 的 `FileMetricLogger` 与新版 `Metric` trait，将每个 batch/epoch 的指标
//! 写入结构化日志文件。
//! 日志目录结构：
//! ```text
//! {log_dir}/train/epoch-{N}/Total_Loss.log
//! {log_dir}/train/epoch-{N}/Policy_Loss.log
//! {log_dir}/train/epoch-{N}/Value_Loss.log
//! {log_dir}/train/epoch-{N}/Entropy.log
//! {log_dir}/train/epoch-{N}/Explained_Variance.log
//! ```
//!
//! 每个指标文件按 batch 追加一行（`value,count`，按 batch 大小加权），
//! epoch 末尾追加最终值行（`value,final`），聚合时直接采用该最终值。
//! 训练结束后可以用 `burn::train::LearnerSummary` 读取并打印 Min/Max 汇总表。

use std::path::Path;
use std::sync::Arc;

use burn::data::dataloader::Progress;
use burn::optim::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn::train::logger::{FileMetricLogger, MetricLogger};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::store::{MetricsUpdate, Split};
use burn::train::metric::{
    Metric, MetricAttributes, MetricDefinition, MetricEntry, MetricId, MetricMetadata, MetricName,
    Numeric, NumericAttributes, NumericEntry, SerializedEntry,
};

// ── 指标 ID ──

const METRIC_TOTAL_LOSS: &str = "Total Loss";
const METRIC_POLICY_LOSS: &str = "Policy Loss";
const METRIC_VALUE_LOSS: &str = "Value Loss";
const METRIC_ENTROPY: &str = "Entropy";
const METRIC_EXPLAINED_VAR: &str = "Explained Variance";

// ── 自定义标量指标 ──

/// 标量指标的输入：一个值与其覆盖的样本数。
#[derive(Debug, Clone)]
pub struct ScalarInput {
    value: f64,
    count: usize,
}

impl ScalarInput {
    /// 构造输入，`count` 用于加权聚合（如 batch size）。
    pub fn new(value: f64, count: usize) -> Self {
        Self { value, count }
    }
}

/// 通用标量指标（loss、entropy、explained variance 等），内部用 [`NumericMetricState`]
/// 累积带权均值。
#[derive(Clone)]
pub struct ScalarMetric {
    name: Arc<String>,
    higher_is_better: bool,
    state: NumericMetricState,
}

impl ScalarMetric {
    /// 创建指标，`higher_is_better` 描述数值方向（用于定义元信息）。
    pub fn new(name: &str, higher_is_better: bool) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            higher_is_better,
            state: NumericMetricState::new(),
        }
    }
}

impl Metric for ScalarMetric {
    type Input = ScalarInput;

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.state.update(item.value, item.count);
        self.state
            .compute_update(FormatOptions::new(self.name.clone()).precision(4))
    }

    fn compute(&mut self) -> SerializedEntry {
        // 没有任何样本时避免除零（NumericMetricState 内部做 sum / count）
        if matches!(
            self.state.running_value(),
            NumericEntry::Aggregated { count: 0, .. }
        ) {
            return SerializedEntry::not_available(None);
        }
        self.state
            .compute_final(FormatOptions::new(self.name.clone()).precision(4))
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: self.higher_is_better,
        }
        .into()
    }
}

impl Numeric for ScalarMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

// ── 训练指标封装 ──

/// 封装 burn `FileMetricLogger` 与一组自定义标量指标，提供简洁的记录接口。
pub struct TrainingLogger {
    logger: FileMetricLogger,
    total_loss: ScalarMetric,
    policy_loss: ScalarMetric,
    value_loss: ScalarMetric,
    entropy: ScalarMetric,
    explained_var: ScalarMetric,
    total_loss_id: MetricId,
    policy_loss_id: MetricId,
    value_loss_id: MetricId,
    entropy_id: MetricId,
    explained_var_id: MetricId,
}

impl TrainingLogger {
    /// 创建 logger，日志写入 `log_dir` 目录下。
    ///
    /// 指标定义在构造时注册（`FileMetricLogger::log` 要求定义已存在）。
    pub fn new(log_dir: &Path) -> Self {
        let mut logger = FileMetricLogger::new(log_dir);

        let total_loss = ScalarMetric::new(METRIC_TOTAL_LOSS, false);
        let policy_loss = ScalarMetric::new(METRIC_POLICY_LOSS, false);
        let value_loss = ScalarMetric::new(METRIC_VALUE_LOSS, false);
        let entropy = ScalarMetric::new(METRIC_ENTROPY, false);
        let explained_var = ScalarMetric::new(METRIC_EXPLAINED_VAR, true);

        let total_loss_id = MetricId::new(total_loss.name());
        let policy_loss_id = MetricId::new(policy_loss.name());
        let value_loss_id = MetricId::new(value_loss.name());
        let entropy_id = MetricId::new(entropy.name());
        let explained_var_id = MetricId::new(explained_var.name());

        for (id, metric) in [
            (&total_loss_id, &total_loss),
            (&policy_loss_id, &policy_loss),
            (&value_loss_id, &value_loss),
            (&entropy_id, &entropy),
            (&explained_var_id, &explained_var),
        ] {
            logger.log_metric_definition(MetricDefinition::new(id.clone(), metric));
        }

        Self {
            logger,
            total_loss,
            policy_loss,
            value_loss,
            entropy,
            explained_var,
            total_loss_id,
            policy_loss_id,
            value_loss_id,
            entropy_id,
            explained_var_id,
        }
    }

    /// 记录单个 batch 的训练指标。
    ///
    /// 每调用一次即在对应 epoch 的 `.log` 文件中追加一行，
    /// `FileMetricLogger` 在 epoch 变化时自动切换到新文件。
    // 指标字段天然扁平，逐个参数比包装 struct 更直观，故豁免参数数量检查
    #[allow(clippy::too_many_arguments)]
    pub fn log_batch(
        &mut self,
        epoch: usize,
        step: usize,
        batch_size: usize,
        train_size: usize,
        lr: f64,
        total_loss: f32,
        policy_loss: f32,
        value_loss: f32,
        entropy: f32,
    ) {
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: step * batch_size,
                items_total: train_size,
                unit: Some("items".into()),
            },
            iteration: Some(step),
            lr: Some(ModuleLearningRate::from(lr)),
        };

        let entries = vec![
            MetricEntry::new(
                self.total_loss_id.clone(),
                self.total_loss
                    .update(&ScalarInput::new(total_loss as f64, batch_size), &metadata),
            ),
            MetricEntry::new(
                self.policy_loss_id.clone(),
                self.policy_loss
                    .update(&ScalarInput::new(policy_loss as f64, batch_size), &metadata),
            ),
            MetricEntry::new(
                self.value_loss_id.clone(),
                self.value_loss
                    .update(&ScalarInput::new(value_loss as f64, batch_size), &metadata),
            ),
            MetricEntry::new(
                self.entropy_id.clone(),
                self.entropy
                    .update(&ScalarInput::new(entropy as f64, batch_size), &metadata),
            ),
        ];

        self.logger
            .log(MetricsUpdate::new(entries, vec![]), epoch, &Split::Train);
    }

    /// 记录 epoch 级别的汇总指标（explained_variance），
    /// 并写入各指标本 epoch 的最终值（`value,final` 行）。
    pub fn log_epoch_summary(&mut self, epoch: usize, explained_variance: f32) {
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: 0,
                items_total: 0,
                unit: Some("items".into()),
            },
            iteration: None,
            lr: None,
        };

        let entries = vec![
            MetricEntry::new(
                self.explained_var_id.clone(),
                self.explained_var
                    .update(&ScalarInput::new(explained_variance as f64, 1), &metadata),
            ),
            MetricEntry::new(self.total_loss_id.clone(), self.total_loss.compute()),
            MetricEntry::new(self.policy_loss_id.clone(), self.policy_loss.compute()),
            MetricEntry::new(self.value_loss_id.clone(), self.value_loss.compute()),
            MetricEntry::new(self.entropy_id.clone(), self.entropy.compute()),
            MetricEntry::new(self.explained_var_id.clone(), self.explained_var.compute()),
        ];

        // 重置所有指标状态，准备下一个 epoch
        for metric in [
            &mut self.total_loss,
            &mut self.policy_loss,
            &mut self.value_loss,
            &mut self.entropy,
            &mut self.explained_var,
        ] {
            metric.clear();
        }

        self.logger
            .log(MetricsUpdate::new(entries, vec![]), epoch, &Split::Train);
    }
}

// ── 批次统计（用于控制台输出） ──

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
