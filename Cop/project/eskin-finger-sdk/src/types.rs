use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Force3D {
    pub fx: u32,
    pub fy: u32,
    pub fz: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Force3F {
    pub fx: f32,
    pub fy: f32,
    pub fz: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorModule {
    ThumbProximal = 0,
    ThumbMiddle = 1,
    ThumbTip = 2,
    ThumbNail = 3,

    IndexProximal = 4,
    IndexMiddle = 5,
    IndexTip = 6,
    IndexNail = 7,

    MiddleProximal = 8,
    MiddleMiddle = 9,
    MiddleTip = 10,
    MiddleNail = 11,

    RingProximal = 12,
    RingMiddle = 13,
    RingTip = 14,
    RingNail = 15,

    PinkyProximal = 16,
    PinkyMiddle = 17,
    PinkyTip = 18,
    PinkyNail = 19,

    Palm1 = 20,
    Palm2 = 21,
    Palm3 = 22,
    Palm4 = 23,
    Palm5 = 24,
    Palm6 = 25,
    Palm7 = 26,
    Palm8 = 27,
}

impl SensorModule {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => Self::ThumbProximal,
            1 => Self::ThumbMiddle,
            2 => Self::ThumbTip,
            3 => Self::ThumbNail,
            4 => Self::IndexProximal,
            5 => Self::IndexMiddle,
            6 => Self::IndexTip,
            7 => Self::IndexNail,
            8 => Self::MiddleProximal,
            9 => Self::MiddleMiddle,
            10 => Self::MiddleTip,
            11 => Self::MiddleNail,
            12 => Self::RingProximal,
            13 => Self::RingMiddle,
            14 => Self::RingTip,
            15 => Self::RingNail,
            16 => Self::PinkyProximal,
            17 => Self::PinkyMiddle,
            18 => Self::PinkyTip,
            19 => Self::PinkyNail,
            20 => Self::Palm1,
            21 => Self::Palm2,
            22 => Self::Palm3,
            23 => Self::Palm4,
            24 => Self::Palm5,
            25 => Self::Palm6,
            26 => Self::Palm7,
            27 => Self::Palm8,
            _ => Self::ThumbProximal,
        }
    }
}

impl From<u32> for SensorModule {
    fn from(value: u32) -> Self {
        match value {
            0x1000 => SensorModule::ThumbProximal,
            0x1200 => SensorModule::ThumbMiddle,
            0x1400 => SensorModule::ThumbTip,
            0x1600 => SensorModule::ThumbNail,
            0x1800 => SensorModule::IndexProximal,
            0x1A00 => SensorModule::IndexMiddle,
            0x1C00 => SensorModule::IndexTip,
            0x1E00 => SensorModule::IndexNail,
            0x2000 => SensorModule::MiddleProximal,
            0x2200 => SensorModule::MiddleMiddle,
            0x2400 => SensorModule::MiddleTip,
            0x2600 => SensorModule::MiddleNail,
            0x2800 => SensorModule::RingProximal,
            0x2A00 => SensorModule::RingMiddle,
            0x2C00 => SensorModule::RingTip,
            0x2E00 => SensorModule::RingNail,
            0x3000 => SensorModule::PinkyProximal,
            0x3200 => SensorModule::PinkyMiddle,
            0x3400 => SensorModule::PinkyTip,
            0x3600 => SensorModule::PinkyNail,
            _ => SensorModule::IndexTip,
        }
    }
}


pub const SENSOR_MODULE_COUNT: usize = 28;

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionForce {
    pub module: SensorModule,
    pub point_count: u16,
    pub points: Vec<ForcePoint>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CombinedForce {
    pub module: SensorModule,
    pub force: Force3D,
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerSample {
    pub timestamp_us: u64,
    pub sequence: u32,
    pub combined_forces: CombinedForce,
    // pub distribution_forces: Vec<DistributionForce>,
    // pub module_errors: Vec<ModuleError>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ModuleError {
    pub module: u8,
    pub error_code: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ForcePoint {
    pub fx: i8,
    pub fy: i8,
    pub fz: i8,
}
