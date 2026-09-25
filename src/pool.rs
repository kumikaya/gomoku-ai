//! 固定 K 条工作线程的阻塞式线程池。
//!
//! 自对弈与对弈评估都需要让多局棋并发推进，而每一局在每步 MCTS 中都会
//! 阻塞等待一次 GPU 推理回包。把这些局面交给一个 K 条线程的池子，GPU
//! 线程就能在最多 `K` 个待处理请求上攒批（见 `inference` 模块）。
//!
//! 与 async executor 的差别：任务会一直占住一条工作线程直到跑完，不需要
//! future 状态机、唤醒与重新调度。由于工作线程几乎全程阻塞在 channel 上，
//! `K` 可以远大于 CPU 核数——它的实际含义是「同时在途的推理请求数上限」。

use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};

/// 池内任务。返回值的传递由提交方自己通过 channel 完成。
type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct BlockingPool {
    tx: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl BlockingPool {
    /// 创建线程池。
    ///
    /// `num_threads = 0` 表示自动：`available_parallelism() * 4`，并夹在
    /// `[8, 128]` 之间。工作线程绝大多数时间阻塞在推理回包上，所以这个
    /// 数字代表的是期望的在途请求数，而不是 CPU 占用。
    pub fn new(num_threads: usize) -> Self {
        let num_threads = if num_threads == 0 {
            Self::auto_threads()
        } else {
            num_threads
        };

        let (tx, rx) = crossbeam_channel::unbounded::<Job>();
        let workers = (0..num_threads)
            .map(|i| {
                let rx: Receiver<Job> = rx.clone();
                std::thread::Builder::new()
                    .name(format!("gomoku-pool-{i}"))
                    .spawn(move || worker_loop(rx))
                    .expect("failed to spawn pool worker thread")
            })
            .collect();

        Self {
            tx: Some(tx),
            workers,
        }
    }

    fn auto_threads() -> usize {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        (cores * 4).clamp(8, 128)
    }

    /// 池内工作线程数。
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }

    /// 提交单个任务，阻塞直到拿到返回值。
    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (result_tx, result_rx) = crossbeam_channel::bounded(1);
        self.dispatch(Box::new(move || {
            let _ = result_tx.send(f());
        }));
        result_rx.recv().expect("pool worker thread panicked")
    }

    /// 批量提交任务，按提交顺序返回结果。
    ///
    /// 所有任务一次性入队，因此最多有 `num_threads` 个任务同时在跑。
    pub fn run_all<F, R, I>(&self, jobs: I) -> Vec<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
        I: IntoIterator<Item = F>,
    {
        let mut receivers = Vec::new();
        for f in jobs {
            let (result_tx, result_rx) = crossbeam_channel::bounded(1);
            self.dispatch(Box::new(move || {
                let _ = result_tx.send(f());
            }));
            receivers.push(result_rx);
        }
        receivers
            .into_iter()
            .map(|rx| rx.recv().expect("pool worker thread panicked"))
            .collect()
    }

    fn dispatch(&self, job: Job) {
        self.tx
            .as_ref()
            .expect("thread pool already shut down")
            .send(job)
            .expect("all pool worker threads have exited");
    }
}

impl Drop for BlockingPool {
    fn drop(&mut self) {
        // 丢弃发送端关闭任务队列，工作线程随即从 recv() 收到 Err 退出
        self.tx.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(rx: Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        // 任务 panic 时保持工作线程存活，避免池容量被静默削减。
        // panic 信息仍由默认 hook 打印，且提交方会因结果 channel 断开而感知到。
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use super::*;

    #[test]
    fn run_returns_value() {
        let pool = BlockingPool::new(2);
        assert_eq!(pool.num_threads(), 2);
        assert_eq!(pool.run(|| 1 + 1), 2);
    }

    #[test]
    fn run_all_preserves_submission_order() {
        let pool = BlockingPool::new(4);
        // 故意让先提交的任务更慢，结果仍应按提交顺序返回
        let results = pool.run_all((0..16).map(|i| {
            move || {
                if i % 3 == 0 {
                    std::thread::sleep(Duration::from_millis(5));
                }
                i * i
            }
        }));
        assert_eq!(results, (0..16).map(|i| i * i).collect::<Vec<_>>());
    }

    #[test]
    fn run_all_bounds_concurrency_to_pool_size() {
        let pool = BlockingPool::new(2);
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        pool.run_all((0..8).map(|_| {
            let running = Arc::clone(&running);
            let peak = Arc::clone(&peak);
            move || {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(10));
                running.fetch_sub(1, Ordering::SeqCst);
            }
        }));

        let peak = peak.load(Ordering::SeqCst);
        assert!(peak <= 2, "并发数 {peak} 超过了池容量 2");
        assert!(peak >= 2, "池容量 2 时应当真的并行，实测并发数 {peak}");
    }

    #[test]
    fn panicking_job_does_not_kill_pool() {
        let pool = BlockingPool::new(2);

        // 静音 panic 输出，避免污染测试日志
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.run(|| panic!("boom"))))
                .is_err();
        std::panic::set_hook(hook);

        assert!(panicked, "任务 panic 应该传递给提交方");
        assert_eq!(pool.num_threads(), 2, "工作线程不应因任务 panic 而退出");
        assert_eq!(pool.run(|| 42), 42);
    }
}
