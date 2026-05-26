use serde::{Deserialize, Serialize};

use crate::error::SdkError;

pub const FRAME_START_REQUEST: [u8; 2] = [0x55, 0xAA];
pub const FRAME_START_RESPONSE: [u8; 2] = [0xAA, 0x55];

pub const FUNC_READ: u8 = 0xFB;
pub const FUNC_WRITE: u8 = 0x79;
pub const FUNC_RESPONSE_READ: u8 = 0xFB;
pub const FUNC_RESPONSE_WRITE: u8 = 0xF9;

pub const FRAME_HEADER_LEN: usize = 13;
pub const FRAME_CRC_LEN: usize = 1;
pub const FRAME_STATUS_LEN: usize = 1;
pub const MIN_RESPONSE_FRAME_LEN: usize = FRAME_HEADER_LEN + FRAME_STATUS_LEN + FRAME_CRC_LEN;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceStatus {
    Success = 0x00,
    ReadLenExceeded = 0x01,
    LengthError = 0x02,
    InvalidAddress = 0x03,
    ReadOnlyRegister = 0x04,
}

impl DeviceStatus {
    pub fn to_error(&self) -> Option<SdkError> {
        match self {
            DeviceStatus::Success => None,
            DeviceStatus::ReadLenExceeded => Some(SdkError::DeviceError(0x0001)),
            DeviceStatus::LengthError => Some(SdkError::DeviceError(0x0002)),
            DeviceStatus::InvalidAddress => Some(SdkError::DeviceError(0x0003)),
            DeviceStatus::ReadOnlyRegister => Some(SdkError::DeviceError(0x0004)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameFunction {
    Read,
    Write,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolFrame {
    pub start: [u8; 2],
    pub device_addr: u8,
    pub function: u8,
    pub start_addr: u32,
    pub payload: Vec<u8>,
    pub status: Option<DeviceStatus>,
}

pub struct ReadRequest {
    pub device_addr: u8,
    pub start_addr: u32,
    pub read_byte_count: u16,
}

pub struct WriteRequest {
    pub device_addr: u8,
    pub start_addr: u32,
    pub data: Vec<u8>,
}

pub struct ReadResponse {
    pub device_addr: u8,
    pub start_addr: u32,
    pub data: Vec<u8>,
    pub status: DeviceStatus,
}

pub struct WriteResponse {
    pub device_addr: u8,
    pub start_addr: u32,
    pub return_byte_count: u16,
    pub status: DeviceStatus,
}

pub trait ProtocolCodec: Send + Sync {
    fn encode_read_request(&self, request: &ReadRequest) -> Result<Vec<u8>, SdkError>;
    fn encode_write_request(&self, request: &WriteRequest) -> Result<Vec<u8>, SdkError>;
    fn decode_read_response(&self, frame: &[u8]) -> Result<ReadResponse, SdkError>;
    fn decode_write_response(&self, frame: &[u8]) -> Result<WriteResponse, SdkError>;
    fn decode_stream_frame(&self, frame: &[u8]) -> Result<ProtocolFrame, SdkError>;
    fn crc8(&self, data: &[u8]) -> u8;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EskinProtocolCodec;

impl EskinProtocolCodec {
    fn status_from_u8(raw: u8) -> Result<DeviceStatus, SdkError> {
        match raw {
            0x00 => Ok(DeviceStatus::Success),
            0x01 => Ok(DeviceStatus::ReadLenExceeded),
            0x02 => Ok(DeviceStatus::LengthError),
            0x03 => Ok(DeviceStatus::InvalidAddress),
            0x04 => Ok(DeviceStatus::ReadOnlyRegister),
            other => Err(SdkError::DeviceError(other as u16)),
        }
    }

    fn validate_crc(&self, frame: &[u8]) -> Result<(), SdkError> {
        if frame.len() < FRAME_CRC_LEN {
            return Err(SdkError::FrameError("frame too short for crc".into()));
        }

        let expected = frame[frame.len() - 1];
        let actual = self.crc8(&frame[..frame.len() - 1]);
        if expected != actual {
            return Err(SdkError::CrcError { expected, actual });
        }

        Ok(())
    }

    fn read_u16_le(frame: &[u8], offset: usize) -> Result<u16, SdkError> {
        let bytes = frame
            .get(offset..offset + 2)
            .ok_or_else(|| SdkError::FrameError("missing u16 field".into()))?;

        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32_le(frame: &[u8], offset: usize) -> Result<u32, SdkError> {
        let bytes = frame
            .get(offset..offset + 4)
            .ok_or_else(|| SdkError::FrameError("missing u32 field".into()))?;

        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

impl ProtocolCodec for EskinProtocolCodec {
    fn encode_read_request(&self, request: &ReadRequest) -> Result<Vec<u8>, SdkError> {
        if request.read_byte_count == 0 {
            return Err(SdkError::InvalidParameter(
                "read_byte_count must be greater than 0".into(),
            ));
        }

        let data_len: u16 = 9;
        let mut frame = Vec::with_capacity(14);
        frame.extend_from_slice(&FRAME_START_REQUEST);
        frame.extend_from_slice(&data_len.to_le_bytes());
        frame.push(request.device_addr);
        frame.push(0x00);
        frame.push(FUNC_READ);
        frame.extend_from_slice(&request.start_addr.to_le_bytes());
        frame.extend_from_slice(&request.read_byte_count.to_le_bytes());

        let crc = self.crc8(&frame);
        frame.push(crc);
        debug_println!("send: {:02X?}", frame);
        Ok(frame)
    }

    fn encode_write_request(&self, request: &WriteRequest) -> Result<Vec<u8>, SdkError> {
        if request.data.is_empty() {
            return Err(SdkError::InvalidParameter(
                "write data must not be empty".into(),
            ));
        }

        if request.data.len() > u16::MAX as usize {
            return Err(SdkError::InvalidParameter(
                "write data length exceeds u16::MAX".into(),
            ));
        }

        let write_len = request.data.len() as u16;
        let data_len = 9u16
            .checked_add(write_len)
            .ok_or_else(|| SdkError::InvalidParameter("write frame too large".into()))?;

        let mut frame = Vec::with_capacity(14 + request.data.len());
        frame.extend_from_slice(&FRAME_START_REQUEST);
        frame.extend_from_slice(&data_len.to_le_bytes());
        frame.push(request.device_addr);
        frame.push(0x00);
        frame.push(FUNC_WRITE);
        frame.extend_from_slice(&request.start_addr.to_le_bytes());
        frame.extend_from_slice(&write_len.to_le_bytes());
        frame.extend_from_slice(&request.data);

        let crc = self.crc8(&frame);
        frame.push(crc);

        Ok(frame)
    }

    fn decode_read_response(&self, frame: &[u8]) -> Result<ReadResponse, SdkError> {
        if frame.len() < MIN_RESPONSE_FRAME_LEN {
            return Err(SdkError::FrameError("read response too short".into()));
        }

        // let start = Self::read_u16_le(frame, 0)?;
        let start = &frame[0..2];
        if start != FRAME_START_RESPONSE {
            return Err(SdkError::FrameError(format!(
                "invalid response start: 0x{start:02X?}"
            )));
        }

        debug_println!("get resp");
        let data_len = Self::read_u16_le(frame, 2)? as usize;
        let expected_len = 2 + 2 + 1 + data_len;

        if frame.len() != expected_len {
            return Err(SdkError::FrameError(format!(
                "read response length mismatch: expected {expected_len}, got {}",
                frame.len()
            )));
        }

        self.validate_crc(frame)?;

        let device_addr = frame[4];
        let reserved = frame[5];
        let function = frame[6];

        if reserved != 0x00 {
            return Err(SdkError::FrameError(format!(
                "invalid reserved byte: 0x{reserved:02X}"
            )));
        }

        if function != FUNC_RESPONSE_READ {
            return Err(SdkError::FrameError(format!(
                "invalid read response function: 0x{function:02X}"
            )));
        }

        let start_addr = Self::read_u32_le(frame, 7)?;
        let read_len = Self::read_u16_le(frame, 11)? as usize;
        if data_len != 10 + read_len {
            return Err(SdkError::FrameError(format!(
                "read response data length mismatch: header data_len {data_len}, payload len {read_len}"
            )));
        }

        let payload_start = 14;
        let payload_end = payload_start + read_len;

        let data = frame
            .get(payload_start..payload_end)
            .ok_or_else(|| SdkError::FrameError("read response payload missing".into()))?
            .to_vec();

        let status_offset = 13;
        let status_raw = *frame
            .get(status_offset)
            .ok_or_else(|| SdkError::FrameError("read response status missing".into()))?;
        let status = Self::status_from_u8(status_raw)?;

        if let Some(err) = status.to_error() {
            return Err(err);
        }

        Ok(ReadResponse {
            device_addr,
            start_addr,
            data,
            status,
        })
    }

    fn decode_write_response(&self, frame: &[u8]) -> Result<WriteResponse, SdkError> {
        if frame.len() < MIN_RESPONSE_FRAME_LEN {
            return Err(SdkError::FrameError("write response too short".into()));
        }

        let start = &frame[..2];
        if start != FRAME_START_RESPONSE {
            return Err(SdkError::FrameError(format!(
                "invalid response start: 0x{start:02X?}"
            )));
        }

        let data_len = Self::read_u16_le(frame, 2)? as usize;
        let expected_len = 2 + 2 + 1 + data_len;

        if frame.len() != expected_len {
            return Err(SdkError::FrameError(format!(
                "write response length mismatch: expected {expected_len}, got {}",
                frame.len()
            )));
        }

        self.validate_crc(frame)?;
        let device_addr = frame[4];
        let reserved = frame[5];
        let function = frame[6];

        if reserved != 0x00 {
            return Err(SdkError::FrameError(format!(
                "invalid reserved byte: 0x{reserved:02X}"
            )));
        }

        if function != FUNC_RESPONSE_WRITE {
            return Err(SdkError::FrameError(format!(
                "invalid write response function: 0x{function:02X}"
            )));
        }

        let start_addr = Self::read_u32_le(frame, 7)?;
        let return_byte_count = Self::read_u16_le(frame, 11)?;
        if data_len != 10 {
            return Err(SdkError::FrameError(format!(
                "write response data length mismatch: expected 9, got {data_len}"
            )));
        }

        let status_offset = 13;
        let status_raw = *frame
            .get(status_offset)
            .ok_or_else(|| SdkError::FrameError("write response status missing".into()))?;
        let status = Self::status_from_u8(status_raw)?;

        if let Some(err) = status.to_error() {
            return Err(err);
        }

        Ok(WriteResponse {
            device_addr,
            start_addr,
            return_byte_count,
            status,
        })
    }

    fn decode_stream_frame(&self, frame: &[u8]) -> Result<ProtocolFrame, SdkError> {
        if frame.len() < MIN_RESPONSE_FRAME_LEN {
            return Err(SdkError::FrameError("stream frame too short".into()));
        }

        let start: [u8; 2] = frame[..2].try_into().unwrap();
        if start != FRAME_START_RESPONSE {
            return Err(SdkError::FrameError(format!(
                "invalid stream frame start: 0x{start:02X?}"
            )));
        }

        let data_len = Self::read_u16_le(frame, 2)? as usize;
        let expected_len = 2 + 2 + data_len + FRAME_STATUS_LEN + FRAME_CRC_LEN;

        if frame.len() != expected_len {
            return Err(SdkError::FrameError(format!(
                "stream frame length mismatch: expected {expected_len}, got {}",
                frame.len()
            )));
        }

        self.validate_crc(frame)?;

        let device_addr = frame[4];
        let function = frame[6];
        let start_addr = Self::read_u32_le(frame, 7)?;
        let payload_len = Self::read_u16_le(frame, 11)? as usize;
        if data_len != 9 + payload_len {
            return Err(SdkError::FrameError(format!(
                "stream frame data length mismatch: header data_len {data_len}, payload len {payload_len}"
            )));
        }

        let payload_start = 13;
        let payload_end = payload_start + payload_len;

        let payload = frame
            .get(payload_start..payload_end)
            .ok_or_else(|| SdkError::FrameError("stream payload missing".into()))?
            .to_vec();

        let status_offset = 13;
        let status_raw = *frame
            .get(status_offset)
            .ok_or_else(|| SdkError::FrameError("stream status missing".into()))?;
        let status = Self::status_from_u8(status_raw)?;

        Ok(ProtocolFrame {
            start,
            device_addr,
            function,
            start_addr,
            payload,
            status: Some(status),
        })
    }

    fn crc8(&self, _data: &[u8]) -> u8 {
        const X25: crc::Crc<u8> = crc::Crc::<u8>::new(&crc::CRC_8_I_432_1);
        X25.checksum(_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn codec() -> EskinProtocolCodec {
        EskinProtocolCodec
    }

    #[test]
    fn encode_read_request_has_correct_structure() {
        let req = ReadRequest {
            device_addr: 0x34,
            start_addr: 0x1C00,
            read_byte_count: 168
        };

        let frame = codec().encode_read_request(&req).unwrap();
        println!("begin eq frame");
        assert_eq!(frame[0], 0x55);
        assert_eq!(frame[1], 0xAA);

        assert_eq!(frame[2], 0x09);
        assert_eq!(frame[3], 0x00);
        assert_eq!(frame[4], 0x34);
        assert_eq!(frame[5], 0x00);

        assert_eq!(frame[6], 0xFB);

        assert_eq!(frame[7], 0x00);
        assert_eq!(frame[8], 0x1C);
        assert_eq!(frame[9], 0x00);
        assert_eq!(frame[10], 0x00);

        assert_eq!(frame[11], 0xA8);
        assert_eq!(frame[12], 0x00);

        let crc = codec().crc8(&frame[..frame.len() - 1]);
        assert_eq!(frame[frame.len() - 1], crc);
        assert_eq!(frame[13], 0x35);

        assert_eq!(frame.len(), 14);
    }
}