//! The proxy's spend log callback (`_ProxyDBLogger`) over the engine in
//! `litellm-callbacks-spend-logger`, writing each served call's row to the gateway's database,
//! with `NativeSpendLogHost` answering its host calls.
//!
//! The Python proxy's spend log monitor decides when the writer flushes, so the gateway runs its
//! own: a pass every `PROXY_BATCH_WRITE_AT` seconds and its jitter, or at once when a row asks for
//! it, and a last drain at shutdown. A pass that fails keeps its rows at the head of the queue for
//! the next one. Python parks a failed pass in its Redis transaction buffer, which only a config
//! `admit` leaves to Python turns on.

use std::{
    sync::{Arc, Weak},
    time::Duration,
};

use litellm_callbacks::ErasedCallback;
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_spend_logger::{SpendLogCallback, SpendLogParts};
use litellm_config::Environment;
use litellm_redaction::TokenRedactor;
use litellm_redis_store::{Remote, spend_counters::SpendCounters};
use litellm_repositories::{Database, KeyRepository, SpendLogRepository};
use litellm_spend_log_writer::{PgLedgerStore, SpendLogFlusher, SpendLogWriter, WriterSettings};
use rand::Rng;
use tokio::sync::Notify;

use super::{Registered, parallel_request_limiter::REDIS_DEFAULT_TTL};
use crate::{RedisClients, host::NativeSpendLogHost};

/// `PROXY_BATCH_WRITE_AT` when unset.
const BATCH_WRITE_AT: Duration = Duration::from_secs(10);
/// `_BATCH_WRITE_JITTER_MAX_SECONDS`.
const JITTER_MAX_SECONDS: u64 = 5;
/// `MAX_LOGS_PER_INTERVAL`.
const PASS_ROWS: usize = 10_000;
/// How long shutdown waits for the queue to drain before it gives the rest up.
const DRAIN_DEADLINE: Duration = Duration::from_secs(30);

type Writer = SpendLogWriter<PgLedgerStore>;

fn settings(environment: &Environment) -> WriterSettings {
    let defaults = WriterSettings::default();
    let at_least_one = |set: Option<u64>, default: usize| {
        set.map_or(default, |set| {
            usize::try_from(set).unwrap_or(usize::MAX).max(1)
        })
    };
    WriterSettings {
        queue_max_bytes: at_least_one(
            environment.spend_log_queue_max_bytes,
            defaults.queue_max_bytes,
        ),
        write_batch_max_bytes: at_least_one(
            environment.spend_log_write_batch_max_bytes,
            defaults.write_batch_max_bytes,
        ),
        write_batch_max_rows: at_least_one(
            environment.spend_log_write_batch_max_rows,
            defaults.write_batch_max_rows,
        ),
        ..defaults
    }
}

/// The spend log writer the callback queues to, and what wakes its flush loop.
pub struct SpendLogs {
    writer: Arc<Writer>,
    wake: Arc<Notify>,
    interval: Duration,
}

impl SpendLogs {
    /// The callback over `database`, charging spend counters in `redis`'s coordination client
    /// when there is one, and the writer it queues to, whose flush loop runs from here on.
    pub(crate) fn start(
        database: &Database,
        redis: &RedisClients,
        environment: &Environment,
    ) -> (Registered, Arc<Self>) {
        let settings = settings(environment);
        let store = PgLedgerStore::new(
            SpendLogRepository::new(database.clone()),
            settings.transaction_timeout,
        );
        let wake = Arc::new(Notify::new());
        let woken = wake.clone();
        let writer = Arc::new(SpendLogWriter::new(
            SpendLogFlusher::new(store, settings),
            move || woken.notify_one(),
        ));
        let counters = SpendCounters::new(redis.coordination.clone().map(|client| Remote {
            client,
            default_ttl: REDIS_DEFAULT_TTL,
        }));
        let callback = SpendLogCallback::new(SpendLogParts {
            writer: writer.clone(),
            budgets: Arc::new(Budgets::new(Arc::new(counters))),
            tokens: Arc::new(TokenRedactor::new(0, Vec::new())),
        });
        let host = NativeSpendLogHost::new(Some(Arc::new(KeyRepository::new(database.clone()))));
        let logs = Arc::new(Self {
            writer,
            wake,
            interval: environment
                .proxy_batch_write_at
                .map_or(BATCH_WRITE_AT, Duration::from_secs),
        });
        tokio::spawn(flushing(Arc::downgrade(&logs)));
        let callback: Arc<dyn ErasedCallback> = Arc::new(callback);
        (Registered::hosted(callback, Arc::new(host)), logs)
    }

    /// Writes passes until the queue is empty or a pass fails.
    async fn flush(&self) {
        while !self.writer.is_empty() {
            if let Err(error) = Box::pin(self.writer.flush(PASS_ROWS, None)).await {
                tracing::error!(
                    error = %litellm_error::Report::new(&error),
                    queued = self.writer.len(),
                    "Spend tracking - the spend log pass failed, its rows stay queued"
                );
                return;
            }
        }
    }

    /// Writes what is still queued, for up to [`DRAIN_DEADLINE`], once the gateway stops serving.
    pub async fn drain(&self) {
        if tokio::time::timeout(DRAIN_DEADLINE, async {
            self.writer.idle().await;
            self.flush().await;
        })
        .await
        .is_err()
        {
            tracing::error!(
                queued = self.writer.len(),
                "Spend tracking - shutdown gave up on the spend logs still queued"
            );
        }
    }
}

/// The flush loop of `logs`, until the gateway drops them.
async fn flushing(logs: Weak<SpendLogs>) {
    loop {
        let Some((wake, interval)) = logs
            .upgrade()
            .map(|logs| (logs.wake.clone(), logs.interval))
        else {
            return;
        };
        let jitter = Duration::from_secs(rand::thread_rng().gen_range(0..=JITTER_MAX_SECONDS));
        tokio::select! {
            () = tokio::time::sleep(interval + jitter) => {}
            () = wake.notified() => {}
        }
        let Some(logs) = logs.upgrade() else {
            return;
        };
        logs.flush().await;
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::the_proxys_defaults(Environment::default(), (64_000_000, 2_000_000, 100))]
    #[case::the_variables_set(
        Environment {
            spend_log_queue_max_bytes: Some(1_000),
            spend_log_write_batch_max_bytes: Some(500),
            spend_log_write_batch_max_rows: Some(0),
            ..Environment::default()
        },
        (1_000, 500, 1)
    )]
    fn the_writers_limits_come_from_the_environment(
        #[case] environment: Environment,
        #[case] limits: (usize, usize, usize),
    ) {
        let settings = settings(&environment);
        assert_eq!(
            (
                settings.queue_max_bytes,
                settings.write_batch_max_bytes,
                settings.write_batch_max_rows
            ),
            limits
        );
    }
}
