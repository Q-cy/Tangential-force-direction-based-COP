use chrono::Duration;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
    channel::{ChannelManager, DeviceEvent},
    config::{DeviceConfig, DeviceInfo},
    error::SdkError,
    protocol::{
        EskinProtocolCodec, FRAME_START_RESPONSE, ProtocolCodec,
        ReadRequest, WriteRequest,
    },
    stream::{StreamConfig, StreamController, StreamRuntime},
    transport::{SerialTransport, SharedSerialTransport},
    types::FingerSample,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    Closed,
    Open,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMode {
    Command,
    Streaming,
}

pub struct EskinDeviceInner {
    pub info: DeviceInfo,
    pub config: DeviceConfig,
    pub channels: Arc<ChannelManager>,
    pub state: DeviceState,
    pub mode: DeviceMode,
    pub transport: SharedSerialTransport,
    pub codec: Box<dyn ProtocolCodec>,
    stream: Option<StreamRuntime>,
}

impl EskinDeviceInner {
    pub fn new(config: DeviceConfig, transport: Box<dyn SerialTransport>) -> Self {
        let channels = ChannelManager::new(
            config.sample_capacity,
            config.command_capacity,
            config.event_capacity,
            config.drop_policy,
        );

        Self {
            info: DeviceInfo::default(),
            config,
            channels: Arc::new(channels),
            state: DeviceState::Closed,
            mode: DeviceMode::Command,
            transport: Arc::new(Mutex::new(transport)),
            codec: Box::new(EskinProtocolCodec),
            stream: None
        }
    }

    pub fn new_shared(config: DeviceConfig, transport: SharedSerialTransport) -> Self {
        let channels = ChannelManager::new(
            config.sample_capacity,
            config.command_capacity,
            config.event_capacity,
            config.drop_policy,
        );

        Self {
            info: DeviceInfo::default(),
            config,
            channels: Arc::new(channels),
            state: DeviceState::Closed,
            mode: DeviceMode::Command,
            transport,
            codec: Box::new(EskinProtocolCodec),
            stream: None
        }
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
            debug_println!("[device] read_exact: need {} bytes, have {} so far, remaining timeout: {:?}", buf.len() - offset, offset, remaining);
            let n = transport.read(&mut buf[offset..], Duration::from_std(remaining).unwrap())?;
            debug_println!("[device] read_exact: got {} bytes: {:02X?}", n, &buf[offset..offset + n]);

            if n == 0 {
                return Err(SdkError::Timeout);
            }

            offset += n;
        }

        Ok(())
    }

    fn read_response_frame_from(
        &self,
        transport: &mut dyn SerialTransport,
    ) -> Result<Vec<u8>, SdkError> {
        let timeout = Duration::milliseconds(self.config.read_timeout_ms as i64);
        let mut header = [0u8; 4];
        Self::read_exact_from_transport(transport, &mut header, timeout)?;
        debug_println!("[device] recv header: {:02X?}", header);

        // let start = u16::from_be_bytes([header[0], header[1]]);
        let start = [header[0], header[1]];
        if start != FRAME_START_RESPONSE {
            return Err(SdkError::FrameError(format!(
                "invalid response start: 0x{start:02X?}"
            )));
        }
        debug_println!("h2: {:02X}, h3: {:02X}", header[2], header[3]);
        let data_len = u16::from_le_bytes([header[2], header[3]]) as usize;
        debug_println!("data_len: {data_len}");
        let total_len = 5 + data_len;
        // let total_len = data_len + 1;

        let mut frame = vec![0u8; total_len];
        frame[..4].copy_from_slice(&header);
        Self::read_exact_from_transport(transport, &mut frame[4..], timeout)?;
        debug_println!("[device] recv frame: {:02X?}", frame);

        Ok(frame)
    }

    fn lock_transport(&self) -> Result<MutexGuard<'_, Box<dyn SerialTransport>>, SdkError> {
        self.transport
            .lock()
            .map_err(|_| SdkError::InternalError("transport mutex poisoned".into()))
    }

    fn ensure_open(&self) -> Result<(), SdkError> {
        match self.state {
            DeviceState::Open => Ok(()),
            DeviceState::Closed => Err(SdkError::NotInitialized),
            DeviceState::Error => Err(SdkError::InternalError("device is in error state".into())),
        }
    }

    fn ensure_command_mode(&self) -> Result<(), SdkError> {
        match self.mode {
            DeviceMode::Command => Ok(()),
            DeviceMode::Streaming => Err(SdkError::StreamingBusy)
        }
    }

    pub fn channels(&self) -> Arc<ChannelManager> {
        Arc::clone(&self.channels)
    }

    pub fn create_stream_runtime(&self) -> StreamRuntime {
        StreamRuntime::new(Arc::clone(&self.channels), Arc::clone(&self.transport))
    }

    pub fn shared_transport(&self) -> SharedSerialTransport {
        Arc::clone(&self.transport)
    }


}

pub trait EskinDeviceFunc {
    fn read_hdw_version(&mut self) -> Result<String, SdkError>;
    fn read_matrix_row(&mut self) -> Result<u8, SdkError>;
    fn read_matrix_col(&mut self) -> Result<u8, SdkError>;
    fn read_device_config1(&mut self) -> Result<u8, SdkError>;
    fn read_device_config2(&mut self) -> Result<u8, SdkError>;
    fn write_device_config1(&mut self, enable: bool) -> Result<u16, SdkError>;
    fn write_device_config2(&mut self, enable: bool) -> Result<u16, SdkError>;
    fn write_matrix_row(&mut self, row: u8) -> Result<u16, SdkError>;
    fn write_matrix_col(&mut self, col: u8) -> Result<u16, SdkError>; 
}

impl EskinDeviceFunc for EskinDeviceInner {
    fn read_hdw_version(&mut self) -> Result<String, SdkError> {
        let hdw = self.read_register(0, 2)
            .map_err(|_| SdkError::FrameError("read hardware version failed".into()))?;

        let version = format!("{}.{}", hdw[0], hdw[1]);
        Ok(version)
    }

    fn read_matrix_row(&mut self) -> Result<u8, SdkError> {
        let row = self.read_register(0x0015, 1)
            .map_err(|_| SdkError::FrameError("read matrix row failed".into()))?;

        Ok(row[0])
    }

    fn read_matrix_col(&mut self) -> Result<u8, SdkError> {
        let col = self.read_register(0x0014, 1)
            .map_err(|_| SdkError::FrameError("read matrix col failed".into()))?;

        Ok(col[0])
    }

    fn write_matrix_row(&mut self, row: u8) -> Result<u16, SdkError> {
        let res = self.write_register(0x0015, &[row])
            .map_err(|_| SdkError::FrameError("write matrix row failed".into()))?;
        Ok(res)
    }

    fn write_matrix_col(&mut self, col: u8) -> Result<u16, SdkError> {
        let res = self.write_register(0x0015, &[col])
            .map_err(|_| SdkError::FrameError("write matrix row failed".into()))?;
        Ok(res)
    }

    fn read_device_config1(&mut self) -> Result<u8, SdkError> {
        let enabled = self.read_register(0x0017, 1)
            .map_err(|_| SdkError::FrameError("read device config1 failed".into()))?;
        Ok(enabled[0])
    }

    fn read_device_config2(&mut self) -> Result<u8, SdkError> {
        let enabled = self.read_register(0x0018, 1)
            .map_err(|_| SdkError::FrameError("read device config2 failed".into()))?;
        Ok(enabled[0])
    }

    fn write_device_config1(&mut self, enable: bool) -> Result<u16, SdkError> {
        self.write_register(0x0017, &[u8::from(enable)])
            .map_err(|_| SdkError::FrameError("write device config1 failed".into()))
    }

    fn write_device_config2(&mut self, enable: bool) -> Result<u16, SdkError> {
        self.write_register(0x0018, &[u8::from(enable)])
            .map_err(|_| SdkError::FrameError("write device config2 failed".into()))
    }
}

pub trait EskinDevice {
    fn open(&mut self) -> Result<(), SdkError>;
    fn close(&mut self) -> Result<(), SdkError>;
    fn state(&self) -> DeviceState;
    fn mode(&self) -> DeviceMode;
    fn device_info(&self) -> Result<DeviceInfo, SdkError>;
    fn config(&self) -> &DeviceConfig;
    fn apply_config(&mut self, config: DeviceConfig) -> Result<(), SdkError>;
    fn start_stream(&mut self) -> Result<(), SdkError>;
    fn stop_stream(&mut self) -> Result<(), SdkError>;
    fn read_sample(&self, timeout_ms: u32) -> Result<FingerSample, SdkError>;
    fn read_event(&self, timeout_ms: u32) -> Result<DeviceEvent, SdkError>;
    fn read_register(&mut self, addr: u32, length: u16) -> Result<Vec<u8>, SdkError>;
    fn write_register(&mut self, addr: u32, data: &[u8]) -> Result<u16, SdkError>;
}

impl EskinDevice for EskinDeviceInner {
    fn open(&mut self) -> Result<(), SdkError> {
        {
            let mut transport = self.lock_transport()?;
            transport.open()?;
            transport.flush_rx()?;
        }
        self.state = DeviceState::Open;
        Ok(())
    }

    fn close(&mut self) -> Result<(), SdkError> {
        if self.mode == DeviceMode::Streaming {
            self.stop_stream()?;
        }
        {
            let mut transport = self.lock_transport()?;
            transport.close()?;
        }
        self.state = DeviceState::Closed;
        Ok(())
    }

    fn state(&self) -> DeviceState {
        self.state
    }

    fn mode(&self) -> DeviceMode {
        self.mode
    }

    fn device_info(&self) -> Result<DeviceInfo, SdkError> {
        Ok(self.info.clone())
    }

    fn config(&self) -> &DeviceConfig {
        &self.config
    }

    fn apply_config(&mut self, config: DeviceConfig) -> Result<(), SdkError> {
        self.config = config;
        Ok(())
    }

    fn start_stream(&mut self) -> Result<(), SdkError> {
        self.ensure_open()?;

        if self.mode == DeviceMode::Streaming {
            return Err(SdkError::StreamingBusy);
        }

        let stream_config = StreamConfig {
            mode: crate::stream::StreamMode::Polling,
            device_addr: self.config.device_addr,
            read_timeout_ms: self.config.read_timeout_ms,
            ..Default::default()
        };
        println!("stream_config: {:?}", stream_config);
        let mut runtime = self.create_stream_runtime();
        runtime.start(stream_config)?;

        self.stream = Some(runtime);
        self.mode = DeviceMode::Streaming;

        Ok(())
    }

    fn stop_stream(&mut self) -> Result<(), SdkError> {
        if self.mode != DeviceMode::Streaming {
            return Err(SdkError::NotStreaming);
        }

        if let Some(mut runtime) = self.stream.take() {
            // Worker 可能已经因为 I/O 错误自行停止，忽略 NotStreaming
            match runtime.stop() {
                Ok(()) | Err(SdkError::NotStreaming) => {}
                Err(e) => return Err(e),
            }
        }

        self.mode = DeviceMode::Command;
        Ok(())
    }

    fn read_sample(&self, timeout_ms: u32) -> Result<FingerSample, SdkError> {
        self.channels.recv_sample(timeout_ms)
    }

    fn read_event(&self, timeout_ms: u32) -> Result<DeviceEvent, SdkError> {
        self.channels.recv_event(timeout_ms)
    }

    fn read_register(&mut self, addr: u32, length: u16) -> Result<Vec<u8>, SdkError> {
        self.ensure_open()?;
        self.ensure_command_mode()?;
        let request = ReadRequest {
            device_addr: self.config.device_addr,
            start_addr: addr,
            read_byte_count: length,
        };

        let request_frame = self.codec.encode_read_request(&request)?;

        let response_frame = {
            let mut transport = self.lock_transport()?;
            transport.flush_rx()?;
            transport.write(&request_frame)?;
            self.read_response_frame_from(transport.as_mut())?
        };
        let response = self.codec.decode_read_response(&response_frame)?;

        Ok(response.data)
    }

    fn write_register(&mut self, addr: u32, data: &[u8]) -> Result<u16, SdkError> {
        self.ensure_open()?;
        self.ensure_command_mode()?;
        let request = WriteRequest {
            device_addr: self.config.device_addr,
            start_addr: addr,
            data: data.to_vec(),
        };

        let request_frame = self.codec.encode_write_request(&request)?;

        let response_frame = {
            let mut transport = self.lock_transport()?;
            transport.flush_rx()?;
            transport.write(&request_frame)?;
            self.read_response_frame_from(transport.as_mut())?
        };
        let response = self.codec.decode_write_response(&response_frame)?;

        Ok(response.return_byte_count)
    }
}
