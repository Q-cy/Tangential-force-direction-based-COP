use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_addr: u8,
    pub auto_distribution: bool,
    pub read_distribution: bool,
    pub drop_policy: DropPolicy,
    pub sample_capacity: usize,
    pub command_capacity: usize,
    pub event_capacity: usize,
    pub read_timeout_ms: u32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_addr: 0x34,
            auto_distribution: false,
            read_distribution: true,
            drop_policy: DropPolicy::DropOldest,
            sample_capacity: 1024,
            command_capacity: 64,
            event_capacity: 128,
            read_timeout_ms: 1000,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Copy)]
pub enum DropPolicy {
    DropNewest,
    DropOldest,
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub serial_number: u32,
    pub firmware_version: u16,
    pub calibration_group: u16,
    pub module_active_status: u16,
    pub l_line: u16,
    pub h_line: u16,
    pub product_config_1: u32,
    pub product_config_2: u32,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            serial_number: 0x0001,
            firmware_version: 0x01,
            calibration_group: 0,
            module_active_status: 0,
            l_line: 7,
            h_line: 12,
            product_config_1: 0,
            product_config_2: 0,
        }
    }
}
