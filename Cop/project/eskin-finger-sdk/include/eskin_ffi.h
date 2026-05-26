#ifndef ESkin_FFI_H
#define ESkin_FFI_H

#include <cstdint>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* EskinDeviceHandle;

typedef struct {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
} EskinSdkVersion;

typedef enum {
    ESkinSuccess = 0,
    ESkinInvalidPointer = 1,
    ESkinDeviceNotFound = 2,
    ESkinDeviceAlreadyOpen = 3,
    ESkinNotInitialized = 4,
    ESkinAlreadyStreaming = 5,
    ESkinNotStreaming = 6,
    ESkinConfigError = 7,
    ESkinIoError = 8,
    ESkinTimeout = 9,
    ESkinChannelClosed = 10,
    ESkinInternalError = 11,
    ESkinBufferOverflow = 12,
    ESkinInvalidParameter = 13,
    ESkinCrcError = 14,
    ESkinFrameError = 15,
    ESkinProtocolError = 16,
    ESkinDeviceError = 17,
} EskinSdkErrorCode;

EskinSdkVersion eskin_version(void);

EskinDeviceHandle eskin_open(const char* path, const void* config);
EskinSdkErrorCode eskin_close(EskinDeviceHandle handle);

EskinSdkErrorCode eskin_read_register(
    EskinDeviceHandle handle,
    uint32_t addr,
    uint16_t length,
    uint8_t* buf,
    uint32_t buf_len,
    uint32_t* actual_len
);

EskinSdkErrorCode eskin_write_register(
    EskinDeviceHandle handle,
    uint32_t addr,
    const uint8_t* data,
    uint16_t data_len,
    uint16_t* return_count
);

// Device function interfaces (EskinDeviceFunc)

EskinSdkErrorCode eskin_read_hdw_version(
    EskinDeviceHandle handle,
    char* buf,
    uint32_t buf_len,
    uint32_t* actual_len
);

EskinSdkErrorCode eskin_read_matrix_row(
    EskinDeviceHandle handle,
    uint8_t* out
);

EskinSdkErrorCode eskin_read_matrix_col(
    EskinDeviceHandle handle,
    uint8_t* out
);

EskinSdkErrorCode eskin_read_device_config1(
    EskinDeviceHandle handle,
    uint8_t* out
);

EskinSdkErrorCode eskin_read_device_config2(
    EskinDeviceHandle handle,
    uint8_t* out
);

EskinSdkErrorCode eskin_write_device_config1(
    EskinDeviceHandle handle,
    bool enable,
    uint16_t* return_count
);

EskinSdkErrorCode eskin_write_device_config2(
    EskinDeviceHandle handle,
    bool enable,
    uint16_t* return_count
);

EskinSdkErrorCode eskin_write_matrix_row(
    EskinDeviceHandle handle,
    uint8_t row,
    uint16_t* return_count
);

EskinSdkErrorCode eskin_write_matrix_col(
    EskinDeviceHandle handle,
    uint8_t col,
    uint16_t* return_count
);

// Streaming interfaces

typedef struct {
    uint32_t fx;
    uint32_t fy;
    uint32_t fz;
} CForce3D;

typedef struct {
    uint32_t module;
    CForce3D force;
} CCombinedForce;

typedef struct {
    uint64_t timestamp_us;
    uint32_t sequence;
    CCombinedForce combined_force;
} CFingerSample;

EskinSdkErrorCode eskin_start_stream(EskinDeviceHandle handle);
EskinSdkErrorCode eskin_stop_stream(EskinDeviceHandle handle);
EskinSdkErrorCode eskin_read_sample(EskinDeviceHandle handle, uint32_t timeout_ms, CFingerSample* out);
EskinSdkErrorCode eskin_get_mode(EskinDeviceHandle handle, uint32_t* out);

#ifdef __cplusplus
}
#endif

#endif
