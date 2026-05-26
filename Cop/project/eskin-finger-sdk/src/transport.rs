use crate::error::SdkError;
use chrono::Duration;
use serialport::{ClearBuffer, DataBits, FlowControl, Parity, StopBits};
use std::io::ErrorKind;
use std::sync::{Arc, Mutex};

pub type SharedSerialTransport = Arc<Mutex<Box<dyn SerialTransport>>>;

pub trait SerialTransport: Send {
    fn open(&mut self) -> Result<(), SdkError>;
    fn close(&mut self) -> Result<(), SdkError>;
    fn is_open(&self) -> bool;
    fn write(&mut self, data: &[u8]) -> Result<usize, SdkError>;
    fn read(&mut self, buf: &mut [u8], timeout: Duration) -> Result<usize, SdkError>;
    fn flush_rx(&mut self) -> Result<(), SdkError>;
}

pub struct SerialPortTransport {
    pub path: String,
    pub baud_rate: u32,
    pub port: Option<Box<dyn serialport::SerialPort>>,
}

impl SerialPortTransport {
    pub fn new(path: impl Into<String>, baud_rate: u32) -> Self {
        Self {
            path: path.into(),
            baud_rate,
            port: None,
        }
    }

    fn port_mut(&mut self) -> Result<&mut Box<dyn serialport::SerialPort>, SdkError> {
        self.port
            .as_mut()
            .ok_or_else(|| SdkError::DeviceNotFound(self.path.clone()))
    }

    fn timeout_to_std(timeout: Duration) -> Result<std::time::Duration, SdkError> {
        timeout
            .to_std()
            .map_err(|_| SdkError::InvalidParameter("timeout must be non-negative".into()))
    }

    fn map_serial_error(error: serialport::Error) -> SdkError {
        SdkError::IoError(std::io::Error::new(ErrorKind::Other, error.to_string()))
    }

    fn map_io_error(error: std::io::Error) -> SdkError {
        match error.kind() {
            ErrorKind::TimedOut | ErrorKind::WouldBlock => SdkError::Timeout,
            _ => SdkError::IoError(error),
        }
    }
}

impl SerialTransport for SerialPortTransport {
    fn open(&mut self) -> Result<(), SdkError> {
        if self.port.is_some() {
            return Err(SdkError::DeviceAlreadyOpen);
        }

        let port = serialport::new(&self.path, self.baud_rate)
            .data_bits(DataBits::Eight)
            .stop_bits(StopBits::One)
            .parity(Parity::None)
            .flow_control(FlowControl::None)
            .open()
            .map_err(Self::map_serial_error)?;

        self.port = Some(port);
        Ok(())
    }

    fn close(&mut self) -> Result<(), SdkError> {
        self.port.take();
        Ok(())
    }

    fn is_open(&self) -> bool {
        self.port.is_some()
    }

    fn write(&mut self, data: &[u8]) -> Result<usize, SdkError> {
        if data.is_empty() {
            return Ok(0);
        }

        let port = self.port_mut()?;
        let written = port.write(data).map_err(Self::map_io_error)?;
        port.flush().map_err(Self::map_io_error)?;

        Ok(written)
    }

    fn read(&mut self, buf: &mut [u8], timeout: Duration) -> Result<usize, SdkError> {
        if buf.is_empty() {
            return Ok(0);
        }

        let timeout = Self::timeout_to_std(timeout)?;
        let port = self.port_mut()?;

        port.set_timeout(timeout).map_err(Self::map_serial_error)?;
        port.read(buf).map_err(Self::map_io_error)
    }

    fn flush_rx(&mut self) -> Result<(), SdkError> {
        self.port_mut()?
            .clear(ClearBuffer::Input)
            .map_err(Self::map_serial_error)
    }
}
