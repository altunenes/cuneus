#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CuneusInstance CuneusInstance;

typedef enum CuneusStatus {
    CUNEUS_STATUS_OK = 0,
    CUNEUS_STATUS_NULL = 1,
    CUNEUS_STATUS_INVALID_ARGUMENT = 2,
    CUNEUS_STATUS_NOT_FOUND = 3,
    CUNEUS_STATUS_IO_ERROR = 4,
} CuneusStatus;

typedef enum CuneusParamType {
    CUNEUS_PARAM_F32 = 0,
    CUNEUS_PARAM_COLOR3 = 1,
} CuneusParamType;

typedef struct CuneusParamDesc {
    const char* id;
    const char* label;
    CuneusParamType param_type;
    float min_value;
    float max_value;
    float default_value;
    uint32_t flags;
} CuneusParamDesc;

size_t cuneus_bin_count(void);
const char* cuneus_bin_name(size_t index);

CuneusInstance* cuneus_instance_open(const char* bin_name, const char* executable_dir, uint16_t remote_port);
void cuneus_instance_free(CuneusInstance* instance);

const char* cuneus_last_error(void);

size_t cuneus_param_count(CuneusInstance* instance);
CuneusStatus cuneus_param_desc(CuneusInstance* instance, size_t index, CuneusParamDesc* out_desc);
CuneusStatus cuneus_set_param_f32(CuneusInstance* instance, const char* id, float value);
CuneusStatus cuneus_set_param_color3(CuneusInstance* instance, const char* id, float r, float g, float b);
CuneusStatus cuneus_pulse(CuneusInstance* instance, float velocity);
CuneusStatus cuneus_note(CuneusInstance* instance, float pitch, float velocity);
CuneusStatus cuneus_set_transport(CuneusInstance* instance, float bpm, float beat, float measure);

#ifdef __cplusplus
}
#endif
