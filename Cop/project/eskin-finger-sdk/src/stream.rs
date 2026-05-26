use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::{config, register::{REG_COMBINED_FORCE, REG_MODULE_ERROR}};
use chrono::Duration;

use crate::{
    channel::{ChannelManager, DeviceEvent},
    error::SdkError,
    protocol::{EskinProtocolCodec, ProtocolCodec},
    transport::{SerialTransport, SharedSerialTransport},
    types::{FingerSample, SensorModule},
};

use crate::protocol::{FRAME_CRC_LEN, FRAME_START_RESPONSE, FRAME_STATUS_LEN, ReadRequest};
use std::thread::{self, JoinHandle};
use std::time::Duration as StdDuration;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    Polling,
    AutoDistribution,
}

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub mode: StreamMode,
    pub read_distribution: bool,
    pub modules: Vec<SensorModule>,
    pub poll_interval_ms: u32,
    pub device_addr: u8,
    pub read_timeout_ms: u32,
    pub finger_addr: u32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            mode: StreamMode::Polling,
            read_distribution: true,
            modules: Vec::new(),
            poll_interval_ms: 10,
            device_addr: 0x34,
            read_timeout_ms: 100,
            finger_addr: 0x1C00
        }
    }
}

pub trait StreamController: Send {
    fn start(&mut self, config: StreamConfig) -> Result<(), SdkError>;
    fn stop(&mut self) -> Result<(), SdkError>;
    fn is_running(&self) -> bool;
    fn next_sample(&self, timeout_ms: u32) -> Result<FingerSample, SdkError>;
    fn next_event(&self, timeout_ms: u32) -> Result<DeviceEvent, SdkError>;
}

pub struct StreamRuntime {
    running: Arc<AtomicBool>,
    config: Option<StreamConfig>,
    channels: Arc<ChannelManager>,
    transport: SharedSerialTransport,
    worker: Option<JoinHandle<()>>,
}

impl StreamRuntime {
    pub fn new(channels: Arc<ChannelManager>, transport: SharedSerialTransport) -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            config: None,
            channels,
            transport,
            worker: None,
        }
    }

    pub fn running_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.running)
    }

    pub fn config(&self) -> Option<&StreamConfig> {
        self.config.as_ref()
    }

    pub fn transport(&self) -> SharedSerialTransport {
        Arc::clone(&self.transport)
    }
}

impl StreamController for StreamRuntime {
    fn start(&mut self, config: StreamConfig) -> Result<(), SdkError> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(SdkError::AlreadyStreaming);
        }

        let collector = make_collector(&config, Arc::clone(&self.transport));
        let worker = StreamWorker::new(
            Arc::clone(&self.running),
            Arc::clone(&self.channels),
            config.clone(),
            collector,
        )
        .spawn();

        self.worker = Some(worker);
        self.config = Some(config);
        self.channels.send_event(DeviceEvent::StreamStarted)?;

        Ok(())
    }

    fn stop(&mut self) -> Result<(), SdkError> {
        if !self.running.swap(false, Ordering::SeqCst) {
            return Err(SdkError::NotStreaming);
        }

        self.channels.send_event(DeviceEvent::StreamStopped)?;

        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| SdkError::InternalError("stream worker panicked".into()))?;
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn next_sample(&self, timeout_ms: u32) -> Result<FingerSample, SdkError> {
        self.channels.recv_sample(timeout_ms)
    }

    fn next_event(&self, timeout_ms: u32) -> Result<DeviceEvent, SdkError> {
        self.channels.recv_event(timeout_ms)
    }
}

pub struct StreamWorker {
    running: Arc<AtomicBool>,
    channels: Arc<ChannelManager>,
    config: StreamConfig,
    collector: Box<dyn SampleCollector>,
}

impl StreamWorker {
    pub fn new(
        running: Arc<AtomicBool>,
        channels: Arc<ChannelManager>,
        config: StreamConfig,
        collector: Box<dyn SampleCollector>,
    ) -> Self {
        Self {
            running,
            channels,
            config,
            collector,
        }
    }

    pub fn spawn(self) -> JoinHandle<()> {
        thread::spawn(move || self.run())
    }

    fn run(mut self) {
        while self.running.load(Ordering::SeqCst) {
            if let Err(err) = self.tick() {
                let _ = self
                    .channels
                    .send_event(DeviceEvent::IoError(err.to_string()));

                self.running.store(false, Ordering::SeqCst);
                break;
            }

            thread::sleep(StdDuration::from_millis(
                self.config.poll_interval_ms as u64,
            ));
        }
    }

    fn tick(&mut self) -> Result<(), SdkError> {
        // let _transport = self
        //     .transport
        //     .lock()
        //     .map_err(|_| SdkError::InternalError("transport mutex poisoned".into()))?;

        let Some(sample) = self.collector.collect_once()? else {
            return Ok(());
        };
        self.channels.send_sample(sample)?;
        Ok(())
        // TODO:
        // 1. encode read request
        // 2. transport.write()
        // 3. transport.read()
        // 4. protocol.decode()
        // 5. register parse
        // 6. channels.send_sample()
    }
}

pub trait SampleCollector: Send {
    fn collect_once(&mut self) -> Result<Option<FingerSample>, SdkError>;
}

pub struct NoopSampleCollector;

impl SampleCollector for NoopSampleCollector {
    fn collect_once(&mut self) -> Result<Option<FingerSample>, SdkError> {
        Ok(None)
    }
}

pub struct PollingSampleCollector {
    transport: SharedSerialTransport,
    codec: Box<dyn ProtocolCodec>,
    config: StreamConfig,
    sequence: u32,
}

impl PollingSampleCollector {
    pub fn new(transport: SharedSerialTransport, config: StreamConfig) -> Self {
        Self {
            transport,
            codec: Box::new(EskinProtocolCodec),
            config,
            sequence: 0,
        }
    }
    fn next_sequence(&mut self) -> u32 {
        let sequence = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        sequence
    }
    fn read_exact_from_transport(
        transport: &mut dyn SerialTransport,
        buf: &mut [u8],
        timeout: Duration,
    ) -> Result<(), SdkError> {
        let deadline = std::time::Instant::now()
            + timeout
                .to_std()
                .map_err(|_| SdkError::InvalidParameter("timeout must be non-negative".into()))?;

        let mut offset = 0;
        while offset < buf.len() {
            let remaining = deadline
                .checked_duration_since(std::time::Instant::now())
                .unwrap_or(std::time::Duration::from_millis(1));
            let n = transport.read(&mut buf[offset..], Duration::from_std(remaining).unwrap())?;

            if n == 0 {
                return Err(SdkError::Timeout);
            }

            offset += n;
        }

        Ok(())
    }
    fn read_response_frame(
        &self,
        transport: &mut dyn SerialTransport,
    ) -> Result<Vec<u8>, SdkError> {
        let timeout = Duration::milliseconds(self.config.read_timeout_ms as i64);

        let mut header = [0u8; 4];
        Self::read_exact_from_transport(transport, &mut header, timeout)?;
        debug_println!("[stream] recv header: {:02X?}", header);

        let start = [header[0], header[1]];
        if start != FRAME_START_RESPONSE {
            return Err(SdkError::FrameError(format!(
                "invalid response start: 0x{start:02X?}"
            )));
        }

        let data_len = u16::from_le_bytes([header[2], header[3]]) as usize;
        let total_len = 4 + data_len + FRAME_CRC_LEN;

        let mut frame = vec![0u8; total_len];
        frame[..4].copy_from_slice(&header);

        Self::read_exact_from_transport(transport, &mut frame[4..], timeout)?;
        debug_println!("[stream] recv frame: {:02X?}", frame);

        Ok(frame)
    }

    fn read_register(&mut self, addr: u32, length: u16) -> Result<Vec<u8>, SdkError> {
        let request = ReadRequest {
            device_addr: self.config.device_addr,
            start_addr: addr,
            read_byte_count: length,
        };

        let request_frame = self.codec.encode_read_request(&request)?;
        let response_frame = {
            let mut transport = self
                .transport
                .lock()
                .map_err(|_| SdkError::InternalError("transport mutex poisoned".into()))?;

            transport.flush_rx()?;
            transport.write(&request_frame)?;
            self.read_response_frame(transport.as_mut())?
        };

        let response = self.codec.decode_read_response(&response_frame)?;
        Ok(response.data)
    }
}

impl SampleCollector for PollingSampleCollector {
    fn collect_once(&mut self) -> Result<Option<FingerSample>, SdkError> {
        let sequence = self.next_sequence();

        let combined_force_raw = self.read_register(self.config.finger_addr, 168)?;

        let combined_forces = crate::register::parse_combined_forces(&combined_force_raw, self.config.finger_addr)?;

        let now = chrono::Utc::now().timestamp_micros() as u64;

        let sample = FingerSample {
            timestamp_us: now,
            sequence,
            combined_forces,
            // distribution_forces: Vec::new(),
            // module_errors
        };

        Ok(Some(sample))
    }
}

fn make_collector(
    config: &StreamConfig,
    transport: SharedSerialTransport,
) -> Box<dyn SampleCollector> {
    match config.mode {
        StreamMode::Polling => Box::new(PollingSampleCollector::new(transport, config.clone())),
        StreamMode::AutoDistribution => Box::new(NoopSampleCollector),
    }
}
