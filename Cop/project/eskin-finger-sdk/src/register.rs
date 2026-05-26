use crate::{
    config::DeviceInfo,
    error::SdkError,
        types::{CombinedForce, DistributionForce, Force3D, ForcePoint, ModuleError, SensorModule},
};

pub const REG_SERIAL_NUMBER: u32 = 0x0000;
pub const REG_FIRMWARE_VERSION: u32 = 0x000F;
pub const REG_CALIBRATION_GROUP: u32 = 0x0010;
pub const REG_MODULE_ACTIVE_STATUS: u32 = 0x0011;
pub const REG_L_LINE: u32 = 0x0012;
pub const REG_H_LINE: u32 = 0x0013;
pub const REG_PRODUCT_CONFIG_1: u32 = 0x0030;
pub const REG_PRODUCT_CONFIG_2: u32 = 0x0032;
pub const REG_COMBINED_FORCE: u32 = 0x1C00;
pub const REG_MODULE_ERROR: u32 = 0x0700;
pub const REG_DISTRIBUTION_FORCE_BASE: u32 = 0x1000;
pub const REG_PROCESSED_VALUE_BASE: u32 = 0x2000;
pub const REG_CALIBRATION_BASE: u32 = 0x8000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterValueType {
    U16,
    U32,
    Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterSpec {
    pub addr: u32,
    pub len: u16,
    pub access: RegisterAccess,
    pub value_type: RegisterValueType,
}

pub const DEVICE_INFO_REGISTERS: &[RegisterSpec] = &[
    RegisterSpec {
        addr: REG_SERIAL_NUMBER,
        len: 4,
        access: RegisterAccess::ReadOnly,
        value_type: RegisterValueType::U32,
    },
    RegisterSpec {
        addr: REG_FIRMWARE_VERSION,
        len: 2,
        access: RegisterAccess::ReadOnly,
        value_type: RegisterValueType::U16,
    },
    RegisterSpec {
        addr: REG_CALIBRATION_GROUP,
        len: 2,
        access: RegisterAccess::ReadOnly,
        value_type: RegisterValueType::U16,
    },
    RegisterSpec {
        addr: REG_MODULE_ACTIVE_STATUS,
        len: 2,
        access: RegisterAccess::ReadOnly,
        value_type: RegisterValueType::U16,
    },
    RegisterSpec {
        addr: REG_L_LINE,
        len: 2,
        access: RegisterAccess::ReadOnly,
        value_type: RegisterValueType::U16,
    },
    RegisterSpec {
        addr: REG_H_LINE,
        len: 2,
        access: RegisterAccess::ReadOnly,
        value_type: RegisterValueType::U16,
    },
    RegisterSpec {
        addr: REG_PRODUCT_CONFIG_1,
        len: 4,
        access: RegisterAccess::ReadWrite,
        value_type: RegisterValueType::U32,
    },
    RegisterSpec {
        addr: REG_PRODUCT_CONFIG_2,
        len: 4,
        access: RegisterAccess::ReadWrite,
        value_type: RegisterValueType::U32,
    },
];

pub trait RegisterMap {
    fn device_info_registers(&self) -> &'static [RegisterSpec];
    fn distribution_register(&self, module: SensorModule) -> Result<RegisterSpec, SdkError>;
    fn parse_device_info(&self, raw: &[u8]) -> Result<DeviceInfo, SdkError>;
    fn parse_distribution_force(
        &self,
        module: SensorModule,
        raw: &[u8],
    ) -> Result<DistributionForce, SdkError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EskinRegisterMap;

impl RegisterMap for EskinRegisterMap {
    fn device_info_registers(&self) -> &'static [RegisterSpec] {
        DEVICE_INFO_REGISTERS
    }

    fn distribution_register(&self, _module: SensorModule) -> Result<RegisterSpec, SdkError> {
        todo!("distribution register spec")
    }

    fn parse_device_info(&self, _raw: &[u8]) -> Result<DeviceInfo, SdkError> {
        todo!("parse device info")
    }

    fn parse_distribution_force(
        &self,
        module: SensorModule,
        raw: &[u8],
    ) -> Result<DistributionForce, SdkError> {
        if raw.len() % 3 != 0 {
            return Err(SdkError::FrameError(format!(
                "distribution force length must be multiple of 3, got {}",
                raw.len()
            )));
        }

        let points = raw
            .chunks_exact(3)
            .map(|chunk| ForcePoint {
                fx: chunk[0] as i8,
                fy: chunk[1] as i8,
                fz: chunk[2] as i8,
            })
            .collect::<Vec<_>>();

        Ok(DistributionForce {
            module,
            point_count: points.len() as u16,
            points,
        })
    }
}


pub fn parse_combined_forces(raw: &[u8], addr: u32) -> Result<CombinedForce, SdkError> {
    // println!("{:02X?}", raw);
    // const MODULE_COUNT: usize = 28;
    // const BYTES_PER_MODULE: usize = 6;

    // if raw.len() < MODULE_COUNT * BYTES_PER_MODULE {
    //     return Err(SdkError::FrameError(format!(
    //         "combined force raw too short: expected {} bytes, got {}",
    //         MODULE_COUNT * BYTES_PER_MODULE,
    //         raw.len()
    //     )));
    // }

    // let mut forces = Vec::with_capacity(MODULE_COUNT);
    // for i in 0..MODULE_COUNT {
    //     let offset =  i * BYTES_PER_MODULE;
    //     let fx = i16::from_le_bytes([raw[offset], raw[offset + 1]]);
    //     let fy = i16::from_le_bytes([raw[offset + 2], raw[offset + 3]]);
    //     let fz = i16::from_le_bytes([raw[offset + 4], raw[offset + 5]]);

    //     forces.push(CombinedForce {
    //         module: SensorModule::from_index(i as u8),
    //         force: Force3D { fx, fy, fz },
    //     });
    // }
    let comb_force: u32 = raw
        .chunks(2)
        .map(|ch| u16::from_le_bytes([ch[0], ch[1]]) as u32)
        .sum();
    let force = CombinedForce {
        module: addr.into(),
        force: Force3D { fx: 0, fy: 0, fz: comb_force }
    };


    Ok(force)
}

pub fn parse_module_errors(raw: &[u8]) -> Result<Vec<ModuleError>, SdkError> {
    const MODULE_COUNT: usize = 28;
    const BYTES_PER_MODULE: usize = 2;

    if raw.len() < MODULE_COUNT * BYTES_PER_MODULE {
        return Err(SdkError::FrameError(format!(
            "module error raw too short: expected {} bytes, got {}",
            MODULE_COUNT * BYTES_PER_MODULE,
            raw.len()
        )));
    }

    let mut errors = Vec::new();
    for i in 0..MODULE_COUNT {
        let offset = i * BYTES_PER_MODULE;
        let error_code = u16::from_le_bytes([raw[offset], raw[offset + 1]]);

        if error_code != 0 {
            errors.push(ModuleError {
                module: i as u8,
                error_code,
            });
        }
    }

    Ok(errors)
}