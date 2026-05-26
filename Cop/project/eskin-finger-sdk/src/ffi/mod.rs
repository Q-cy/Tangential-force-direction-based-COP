use std::{ptr};
use std::ffi::{CStr, c_char};
use crate::device::{DeviceMode, EskinDevice, EskinDeviceFunc};
use crate::transport::SerialPortTransport;
use crate::types::{CombinedForce, FingerSample};
use crate::{config::DeviceConfig, device::EskinDeviceInner, error::SdkErrorCode};

pub type EskinDeviceHandle = *mut core::ffi::c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EskinSdkVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CForce3D {
    pub fx: u32,
    pub fy: u32,
    pub fz: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CCombinedForce {
    pub module: u32,
    pub force: CForce3D,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CFingerSample {
    pub timestamp_us: u64,
    pub sequence: u32,
    pub combined_force: CCombinedForce,
}

#[allow(dead_code)]
struct DeviceWrapper {
    device: EskinDeviceInner,
}

fn sdk_error_to_code(err: crate::error::SdkError) -> SdkErrorCode {
    match err {
        crate::error::SdkError::Timeout => SdkErrorCode::Timeout,
        crate::error::SdkError::FrameError(_) => SdkErrorCode::FrameError,
        crate::error::SdkError::CrcError { .. } => SdkErrorCode::CrcError,
        crate::error::SdkError::DeviceError(_) => SdkErrorCode::DeviceError,
        crate::error::SdkError::IoError(_) => SdkErrorCode::IoError,
        crate::error::SdkError::NotInitialized => SdkErrorCode::NotInitialized,
        crate::error::SdkError::AlreadyStreaming => SdkErrorCode::AlreadyStreaming,
        crate::error::SdkError::NotStreaming => SdkErrorCode::NotStreaming,
        crate::error::SdkError::DeviceNotFound(_) => SdkErrorCode::DeviceNotFound,
        crate::error::SdkError::DeviceAlreadyOpen => SdkErrorCode::DeviceAlreadyOpen,
        crate::error::SdkError::ConfigError(_) => SdkErrorCode::ConfigError,
        crate::error::SdkError::ChannelClosed => SdkErrorCode::ChannelClosed,
        crate::error::SdkError::InternalError(_) => SdkErrorCode::InternalError,
        crate::error::SdkError::BufferOverflow(_) => SdkErrorCode::BufferOverflow,
        crate::error::SdkError::InvalidParameter(_) => SdkErrorCode::InvalidParameter,
        crate::error::SdkError::ProtocolError(_) => SdkErrorCode::ProtocolError,
        crate::error::SdkError::StreamingBusy => SdkErrorCode::StreamingBusy,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn eskin_version() -> EskinSdkVersion {
    EskinSdkVersion { major: 0, minor: 1, patch: 0 }
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_open(
    path: *const c_char,
    config: *const DeviceConfig,
) -> EskinDeviceHandle {
    if path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match unsafe {
        CStr::from_ptr(path)
    }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return ptr::null_mut()
    };

    let device_config = if config.is_null() {
        DeviceConfig::default()
    } else {
        unsafe { (*config).clone() }
    };

    let transport = SerialPortTransport::new(path_str, 921600);
    let mut device = EskinDeviceInner::new(device_config, Box::new(transport));
    
    if device.open().is_err() {
        return ptr::null_mut();
    }

    let wrapper = Box::new(DeviceWrapper {
        device,
    });
    
    Box::into_raw(wrapper) as EskinDeviceHandle
}

/// 关闭设备
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_close(handle: EskinDeviceHandle) -> SdkErrorCode {
    if handle.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.close() {
        Ok(()) => {
            unsafe { drop(Box::from_raw(handle as *mut DeviceWrapper)) };
            SdkErrorCode::Success
        }
        Err(_) => SdkErrorCode::IoError,
    }
}

/// 读寄存器（原始字节）
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_read_register(
    handle: EskinDeviceHandle,
    addr: u32,
    length: u16,
    buf: *mut u8,
    buf_len: u32,
    actual_len: *mut u32,
) -> SdkErrorCode {
    if handle.is_null() || buf.is_null() || actual_len.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    let data = match wrapper.device.read_register(addr, length) {
        Ok(d) => d,
        Err(e) => return sdk_error_to_code(e),
    };

    let copy_len = std::cmp::min(data.len(), buf_len as usize);
    unsafe {
        ptr::copy_nonoverlapping(data.as_ptr(), buf, copy_len);
        *actual_len = data.len() as u32;
    }

    SdkErrorCode::Success
}

/// 写寄存器（原始字节）
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_write_register(
    handle: EskinDeviceHandle,
    addr: u32,
    data: *const u8,
    data_len: u16,
    return_count: *mut u16,
) -> SdkErrorCode {
    if handle.is_null() || data.is_null() || return_count.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len as usize) };

    match wrapper.device.write_register(addr, data_slice) {
        Ok(count) => {
            unsafe { *return_count = count };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 读取硬件版本，写入 buf 中，以 null 结尾
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_read_hdw_version(
    handle: EskinDeviceHandle,
    buf: *mut c_char,
    buf_len: u32,
    actual_len: *mut u32,
) -> SdkErrorCode {
    if handle.is_null() || buf.is_null() || actual_len.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.read_hdw_version() {
        Ok(version) => {
            let bytes = version.as_bytes();
            let copy_len = std::cmp::min(bytes.len(), (buf_len as usize).saturating_sub(1));
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
                *buf.add(copy_len) = 0; // null terminator
                *actual_len = bytes.len() as u32;
            }
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 读取矩阵行数
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_read_matrix_row(
    handle: EskinDeviceHandle,
    out: *mut u8,
) -> SdkErrorCode {
    if handle.is_null() || out.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.read_matrix_row() {
        Ok(row) => {
            unsafe { *out = row };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 读取矩阵列数
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_read_matrix_col(
    handle: EskinDeviceHandle,
    out: *mut u8,
) -> SdkErrorCode {
    if handle.is_null() || out.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.read_matrix_col() {
        Ok(col) => {
            unsafe { *out = col };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 读取设备配置寄存器1
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_read_device_config1(
    handle: EskinDeviceHandle,
    out: *mut u8,
) -> SdkErrorCode {
    if handle.is_null() || out.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.read_device_config1() {
        Ok(val) => {
            unsafe { *out = val };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 读取设备配置寄存器2
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_read_device_config2(
    handle: EskinDeviceHandle,
    out: *mut u8,
) -> SdkErrorCode {
    if handle.is_null() || out.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.read_device_config2() {
        Ok(val) => {
            unsafe { *out = val };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 写入设备配置寄存器1
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_write_device_config1(
    handle: EskinDeviceHandle,
    enable: bool,
    return_count: *mut u16,
) -> SdkErrorCode {
    if handle.is_null() || return_count.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.write_device_config1(enable) {
        Ok(count) => {
            unsafe { *return_count = count };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 写入设备配置寄存器2
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_write_device_config2(
    handle: EskinDeviceHandle,
    enable: bool,
    return_count: *mut u16,
) -> SdkErrorCode {
    if handle.is_null() || return_count.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.write_device_config2(enable) {
        Ok(count) => {
            unsafe { *return_count = count };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 写入矩阵行数
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_write_matrix_row(
    handle: EskinDeviceHandle,
    row: u8,
    return_count: *mut u16,
) -> SdkErrorCode {
    if handle.is_null() || return_count.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.write_matrix_row(row) {
        Ok(count) => {
            unsafe { *return_count = count };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 写入矩阵列数
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_write_matrix_col(
    handle: EskinDeviceHandle,
    col: u8,
    return_count: *mut u16,
) -> SdkErrorCode {
    if handle.is_null() || return_count.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.write_matrix_col(col) {
        Ok(count) => {
            unsafe { *return_count = count };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 启动流式采集
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_start_stream(handle: EskinDeviceHandle) -> SdkErrorCode {
    if handle.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.start_stream() {
        Ok(()) => SdkErrorCode::Success,
        Err(e) => sdk_error_to_code(e),
    }
}

/// 停止流式采集
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_stop_stream(handle: EskinDeviceHandle) -> SdkErrorCode {
    if handle.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.stop_stream() {
        Ok(()) => SdkErrorCode::Success,
        Err(e) => sdk_error_to_code(e),
    }
}

/// 读取一个采样数据（流模式下调用）
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_read_sample(
    handle: EskinDeviceHandle,
    timeout_ms: u32,
    out: *mut CFingerSample,
) -> SdkErrorCode {
    if handle.is_null() || out.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &mut *(handle as *mut DeviceWrapper) };

    match wrapper.device.read_sample(timeout_ms) {
        Ok(sample) => {
            let c_sample = finger_sample_to_c(&sample);
            unsafe { *out = c_sample };
            SdkErrorCode::Success
        }
        Err(e) => sdk_error_to_code(e),
    }
}

/// 查询当前设备模式（Command=0, Streaming=1）
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eskin_get_mode(
    handle: EskinDeviceHandle,
    out: *mut u32,
) -> SdkErrorCode {
    if handle.is_null() || out.is_null() {
        return SdkErrorCode::InvalidPointer;
    }

    let wrapper = unsafe { &*(handle as *const DeviceWrapper) };

    let mode_val = match wrapper.device.mode() {
        DeviceMode::Command => 0u32,
        DeviceMode::Streaming => 1u32,
    };
    unsafe { *out = mode_val };
    SdkErrorCode::Success
}

fn finger_sample_to_c(sample: &FingerSample) -> CFingerSample {
    CFingerSample {
        timestamp_us: sample.timestamp_us,
        sequence: sample.sequence,
        combined_force: combined_force_to_c(&sample.combined_forces),
    }
}

fn combined_force_to_c(cf: &CombinedForce) -> CCombinedForce {
    CCombinedForce {
        module: cf.module as u32,
        force: CForce3D {
            fx: cf.force.fx,
            fy: cf.force.fy,
            fz: cf.force.fz,
        },
    }
}
