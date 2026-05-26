#[repr(C)]
pub enum SdkErrorCode {
    Success = 0,
    InvalidPointer = 1,
    DeviceNotFound = 2,
    DeviceAlreadyOpen = 3,
    NotInitialized = 4,
    AlreadyStreaming = 5,
    NotStreaming = 6,
    ConfigError = 7,
    IoError = 8,
    Timeout = 9,
    ChannelClosed = 10,
    InternalError = 11,
    BufferOverflow = 12,
    InvalidParameter = 13,
    CrcError = 14,
    FrameError = 15,
    ProtocolError = 16,
    DeviceError = 17,
    StreamingBusy = 18,
}

#[derive(Debug, thiserror::Error)]
pub enum SdkError {
    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Device already open")]
    DeviceAlreadyOpen,

    #[error("SDK not initialized")]
    NotInitialized,

    #[error("Already streaming")]
    AlreadyStreaming,

    #[error("Not streaming")]
    NotStreaming,

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Read timeout")]
    Timeout,

    #[error("Channel closed")]
    ChannelClosed,

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Buffer overflow — dropped {0} samples")]
    BufferOverflow(u64),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("CRC error: expected 0x{expected:02X}, got 0x{actual:02X}")]
    CrcError { expected: u8, actual: u8 },

    #[error("Frame error: {0}")]
    FrameError(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("Device error: status 0x{0:04X}")]
    DeviceError(u16),

    #[error("Device is in streaming mode, command not allowed")]
    StreamingBusy
}
