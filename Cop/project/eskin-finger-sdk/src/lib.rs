/// Debug-only print macro. Only outputs when compiled with `--features debug`.
#[cfg(feature = "debug")]
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => { println!($($arg)*) };
}

#[cfg(not(feature = "debug"))]
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {};
}

pub mod channel;
pub mod config;
pub mod device;
pub mod error;
pub mod ffi;
pub mod protocol;
pub mod register;
pub mod stream;
pub mod transport;
pub mod types;
