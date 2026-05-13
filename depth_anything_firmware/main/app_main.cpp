#include "dl_model_base.hpp"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "esp_timer.h"
#include <cmath>
#include <string.h>

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");
extern const uint8_t image_bin_start[] asm("_binary_image_bin_start");
extern const uint8_t image_bin_end[] asm("_binary_image_bin_end");

#define MOUNT_POINT "/sdcard"

static esp_err_t init_sdcard(void)
{
    esp_err_t ret;
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };
    sdmmc_card_t *card;
    const char mount_point[] = MOUNT_POINT;
    
    printf("Initializing SD card using SDMMC peripheral\n");
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    // For ESP32-S3 Korvo-2, standard SDMMC pins might need customization
    // but default often works. If not, users can change pin settings here.
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1; // 1-bit mode is safer on diverse unconfigured boards
    
    ret = esp_vfs_fat_sdmmc_mount(mount_point, &host, &slot_config, &mount_config, &card);
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            printf("Failed to mount filesystem. If you want the card to be formatted, set format_if_mount_failed = true.\n");
        } else {
            printf("Failed to initialize the card (ESP_ERR: %s). Make sure SD card lines have pull-up resistors in place.\n", esp_err_to_name(ret));
        }
        return ret;
    }
    sdmmc_card_print_info(stdout, card);
    return ESP_OK;
}

void run_depth_anything()
{
    printf("Loading model from Flash...\n");
    dl::Model *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);

    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;

    int width = 112;
    int height = 112;
    int channels = 3;
    
    printf("Allocating memory for %d floats...\n", width * height * channels);
    float *input_data = (float *)heap_caps_malloc(width * height * channels * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!input_data) {
        printf("Failed to allocate image buffer.\n");
        delete model;
        return;
    }

    printf("Reading image from Flash...\n");
    size_t expected_size = width * height * channels * sizeof(float);
    size_t image_size = image_bin_end - image_bin_start;
    
    if (image_size != expected_size) {
        printf("Flash image size mismatch. Expected %u bytes, got %u bytes.\n", (unsigned int)expected_size, (unsigned int)image_size);
    } else {
        memcpy(input_data, image_bin_start, expected_size);
        printf("Copied %d floats from flash.\n", width * height * channels);
        printf("Quantizing input data into Model tensor...\n");
        int8_t *input_ptr = (int8_t *)model_input->data;
        int exponent = model_input->exponent; // scale
        for (int i = 0; i < width * height * channels; i++) {
            input_ptr[i] = dl::quantize<int8_t>(input_data[i], DL_RESCALE(exponent));
        }

        printf("Running Model Inference...\n");
        long long start_time = esp_timer_get_time();
        model->run();
        long long end_time = esp_timer_get_time();
        printf("Inference completed in %lld ms\n", (end_time - start_time) / 1000);

        printf("Processing Output Tensor...\n");
        int8_t *output_ptr = (int8_t *)model_output->data;
        int out_exponent = model_output->exponent;

        float min_depth = 999999.0f;
        float max_depth = -999999.0f;

        std::vector<int> out_shape = model_output->get_shape();
        int output_elements = 1;
        for (int dim : out_shape) {
            output_elements *= dim;
        }

        printf("Output tensor shape: ");
        for (int dim : out_shape) {
            printf("%d ", dim);
        }
        printf("\n");

        float* out_floats = (float*)malloc(output_elements * sizeof(float));
        for (int i = 0; i < output_elements; i++) {
            float val = dl::dequantize(output_ptr[i], DL_SCALE(out_exponent));
            out_floats[i] = val;
            if (val < min_depth) min_depth = val;
            if (val > max_depth) max_depth = val;
        }

        printf("\n--- Depth Anything Nano Results ---\n");
        printf("Min Depth: %.4f | Max Depth: %.4f\n", min_depth, max_depth);
        printf("\nCenter 4x4 Output block:\n");
        int center_x = out_shape[2] / 2;
        int center_y = out_shape[1] / 2;
        for (int y = center_y - 2; y < center_y + 2; y++) {
            for (int x = center_x - 2; x < center_x + 2; x++) {
                if (y >= 0 && y < out_shape[1] && x >= 0 && x < out_shape[2]) {
                    printf("%6.2f ", out_floats[y * out_shape[2] + x]);
                }
            }
            printf("\n");
        }
        printf("---------------------------------------------\n");
        
        printf("Skipping writing output depth map to SD card (running from flash).\n");
        
        free(out_floats);
    }
    
    free(input_data);
    delete model;
}

extern "C" void app_main(void)
{
    printf("Starting Depth Anything Nano Inference App\n");
    // Inicializar SD card (no fallar si no hay lector conectado)
    init_sdcard();
    run_depth_anything();
}
