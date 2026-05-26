use crate::{
    config::{DeviceConfig, DropPolicy},
    error::SdkError,
    types::{FingerSample, SensorModule},
};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError, bounded};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub enum DeviceCommand {
    StartStream,
    StopStream,
    SetConfig(DeviceConfig),
    ReadRegister { addr: u32, length: u16 },
    WriteRegister { addr: u32, data: Vec<u8> },
    Shutdown,
}

#[derive(Debug, Clone)]
pub enum DeviceEvent {
    Disconnected(String),
    IoError(String),
    ProtocolError(String),
    ConfigApplied,
    StreamStarted,
    StreamStopped,
    SampleDropped {
        count: u64,
    },
    ModuleError {
        module: SensorModule,
        error_code: u16,
    },
}

pub struct ChannelManager {
    pub sample_tx: Sender<FingerSample>,
    pub sample_rx: Receiver<FingerSample>,

    pub cmd_tx: Sender<DeviceCommand>,
    pub cmd_rx: Receiver<DeviceCommand>,

    pub event_tx: Sender<DeviceEvent>,
    pub event_rx: Receiver<DeviceEvent>,

    pub dropped_samples: AtomicU64,

    pub drop_policy: DropPolicy,
}

impl ChannelManager {
    pub fn new(
        sample_capacity: usize,
        cmd_capacity: usize,
        event_capacity: usize,
        drop_policy: DropPolicy,
    ) -> Self {
        let (sample_tx, sample_rx) = bounded(sample_capacity);
        let (cmd_tx, cmd_rx) = bounded(cmd_capacity);
        let (event_tx, event_rx) = bounded(event_capacity);

        Self {
            sample_tx,
            sample_rx,
            cmd_tx,
            cmd_rx,
            event_tx,
            event_rx,
            dropped_samples: AtomicU64::new(0),
            drop_policy,
        }
    }

    pub fn send_sample(&self, sample: FingerSample) -> Result<(), SdkError> {
        match self.drop_policy {
            DropPolicy::DropNewest => match self.sample_tx.try_send(sample) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(_)) => {
                    self.record_sample_drop();
                    Ok(())
                }
                Err(TrySendError::Disconnected(_)) => Err(SdkError::ChannelClosed),
            },
            DropPolicy::DropOldest => match self.sample_tx.try_send(sample) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(sample)) => {
                    let _ = self.sample_rx.try_recv();
                    self.record_sample_drop();

                    match self.sample_tx.try_send(sample) {
                        Ok(()) => Ok(()),
                        Err(TrySendError::Full(_)) => {
                            self.record_sample_drop();
                            Ok(())
                        }
                        Err(TrySendError::Disconnected(_)) => Err(SdkError::ChannelClosed),
                    }
                }
                Err(TrySendError::Disconnected(_)) => Err(SdkError::ChannelClosed),
            },
        }
    }

    pub fn recv_sample(&self, timeout_ms: u32) -> Result<FingerSample, SdkError> {
        let timeout = std::time::Duration::from_millis(timeout_ms as u64);
        self.sample_rx
            .recv_timeout(timeout)
            .map_err(|err| match err {
                RecvTimeoutError::Timeout => SdkError::Timeout,
                RecvTimeoutError::Disconnected => SdkError::ChannelClosed,
            })
    }

    pub fn send_cmd(&self, cmd: DeviceCommand) -> Result<(), SdkError> {
        self.cmd_tx.try_send(cmd).map_err(|err| match err {
            TrySendError::Full(_) => SdkError::BufferOverflow(1),
            TrySendError::Disconnected(_) => SdkError::ChannelClosed,
        })
    }

    pub fn recv_cmd(&self, timeout_ms: u32) -> Result<DeviceCommand, SdkError> {
        let timeout = std::time::Duration::from_millis(timeout_ms as u64);
        self.cmd_rx.recv_timeout(timeout).map_err(|err| match err {
            RecvTimeoutError::Timeout => SdkError::Timeout,
            RecvTimeoutError::Disconnected => SdkError::ChannelClosed,
        })
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_samples.load(Ordering::Relaxed)
    }

    pub fn reset_dropped_count(&self) {
        self.dropped_samples.store(0, Ordering::Relaxed);
    }

    fn record_sample_drop(&self) -> u64 {
        self.dropped_samples.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn send_event(&self, event: DeviceEvent) -> Result<(), SdkError> {
        self.event_tx.try_send(event).map_err(|err| match err {
            TrySendError::Full(_) => SdkError::BufferOverflow(1),
            TrySendError::Disconnected(_) => SdkError::ChannelClosed,
        })
    }

    pub fn recv_event(&self, timeout_ms: u32) -> Result<DeviceEvent, SdkError> {
        let timeout = std::time::Duration::from_millis(timeout_ms as u64);
        self.event_rx
            .recv_timeout(timeout)
            .map_err(|err| match err {
                RecvTimeoutError::Timeout => SdkError::Timeout,
                RecvTimeoutError::Disconnected => SdkError::ChannelClosed,
            })
    }
}
