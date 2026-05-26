use std::io::{self, BufRead};
use eskin_finger_sdk::{
    config::DeviceConfig,
    device::{EskinDevice, EskinDeviceFunc, EskinDeviceInner},
    error::SdkError,
    transport::SerialPortTransport,
};
fn main() {
    // let transport = SerialPortTransport::new("/dev/ttyUSB0", 921600);
    // let config = DeviceConfig::default();
    // let mut device = EskinDeviceInner::new(config, Box::new(transport));
    // device.open().unwrap();

    // // let data = device.read_register(0x1C00, 168).unwrap();
    // // print_payload_data(&data);

    // read_hdv(&mut device);
    // read_check_group(&mut device);
    // read_row(&mut device);
    // write_col(&mut device, &[0x08]);
    // read_col(&mut device);
    // read_config(&mut device);

    // device.close().unwrap();
    stream_demo();
}

fn read_hdv(device: &mut EskinDeviceInner) {
    let series_id = device.read_register(0, 2).unwrap();
    print_payload_data(&series_id);
}

fn read_check_group(device: &mut EskinDeviceInner) {
    let group = device.read_register(0x000F, 1).unwrap();
    print_payload_data(&group);
}


fn read_row(device: &mut EskinDeviceInner) {
    let row = device.read_register(0x0015, 1).unwrap();
    print_payload_data(&row);
}

fn write_col(device: &mut EskinDeviceInner, col: &[u8]) {
    device.write_register(0x0014, col).unwrap();
}

fn read_col(device: &mut EskinDeviceInner) {
    let col = device.read_register(0x0014, 1).unwrap();
    print_payload_data(&col);
}

fn read_config(device: &mut EskinDeviceInner) {
    let conf = device.read_register(0x0017, 1).unwrap();
    print_payload_data(&conf);
}

/// Stream 模式演示：后台线程持续采集，主线程消费 sample
/// 按 Enter 键停止流式采集
fn stream_demo() {
    let transport = SerialPortTransport::new("/dev/ttyUSB0", 921600);
    let config = DeviceConfig::default();
    let mut device = EskinDeviceInner::new(config, Box::new(transport));
    device.open().unwrap();

    println!("Hardware version: {}", device.read_hdw_version().unwrap());

    // 进入 Streaming 模式
    device.start_stream().unwrap();
    println!("Stream started, mode: {:?}", device.mode());
    println!("Press Enter to stop...");

    // 用 stdin 阻塞线程来检测用户输入，实现优雅退出
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        stdin.lock().lines().next();
        let _ = tx.send(());
    });

    let mut count: u64 = 0;
    loop {
        // 检查用户是否按了 Enter
        if rx.try_recv().is_ok() {
            println!("User requested stop.");
            break;
        }

        match device.read_sample(200) {
            Ok(sample) => {
                count += 1;
                if count % 5 == 0 {
                    println!(
                        "[#{count} seq={}] combined_force={:?}",
                        sample.sequence,
                        sample.combined_forces
                    );
                }
            }
            Err(SdkError::Timeout) => continue,
            Err(e) => {
                eprintln!("read_sample error: {e}");
                break;
            }
        }
    }

    // 回到 Command 模式
    device.stop_stream().unwrap();
    println!("Stream stopped, total samples: {count}, mode: {:?}", device.mode());

    // Stream 停止后，Command 操作恢复正常
    println!("Row: {}", device.read_matrix_row().unwrap());

    device.close().unwrap();
}

fn print_payload_data(data: &[u8]) {
    for (i, chunk) in data.chunks(2).enumerate() {
        if chunk.len() == 2 {
            let val = u16::from_le_bytes([chunk[0], chunk[1]]);
            println!("  [{:3}] [{:02X}] [{:02X}] => {}", i, chunk[0], chunk[1], val);
        } else {
            println!("  [{:3}] [{:02X}] (odd byte)", i, chunk[0]);
        }
    }
}
