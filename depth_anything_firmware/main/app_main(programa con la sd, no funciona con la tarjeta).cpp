#include "dl_model_base.hpp"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "esp_timer.h"
#include <cmath>

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

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

    int input_elements = 3 * 224 * 224;
    printf("Allocating memory for %d floats...\n", input_elements);
    float *img_buffer = (float *)malloc(input_elements * sizeof(float));
    if (!img_buffer) {
        printf("Failed to allocate image buffer.\n");
        delete model;
        return;
    }

    printf("Reading image.bin from SD card...\n");
    FILE *f = fopen(MOUNT_POINT "/image.bin", "rb");
    if (f == NULL) {
        printf("Failed to open image.bin on SD card\n");
        free(img_buffer);
        delete model;
        return;
    }
    size_t read_items = fread(img_buffer, sizeof(float), input_elements, f);
    fclose(f);
    printf("Read %d floats from image.bin\n", read_items);

    if (read_items != input_elements) {
        printf("File size mismatch. Expected %d floats.\n", input_elements);
    } else {
        printf("Quantizing input data into Model tensor...\n");
        int8_t *input_ptr = (int8_t *)model_input->data;
        int exponent = model_input->exponent; // scale
        for (int i = 0; i < input_elements; i++) {
            input_ptr[i] = dl::quantize<int8_t>(img_buffer[i], DL_RESCALE(exponent));
        }

        printf("Running Model Inference...\n");
        long long start_time = esp_timer_get_time();
        model->run();
        long long end_time = esp_timer_get_time();
        printf("Inference completed in %lld ms\n", (end_time - start_time) / 1000);

        printf("Processing Output Tensor...\n");
        int output_elements = 16 * 16;
        int8_t *output_ptr = (int8_t *)model_output->data;
        int out_exponent = model_output->exponent;

        float min_depth = 999999.0f;
        float max_depth = -999999.0f;

        // Extract and dequantize
        float* out_floats = (float*)malloc(output_elements * sizeof(float));
        for (int i = 0; i < output_elements; i++) {
            float val = dl::dequantize(output_ptr[i], DL_SCALE(out_exponent));
            out_floats[i] = val;
            if (val < min_depth) min_depth = val;
            if (val > max_depth) max_depth = val;
        }

        printf("\n--- Depth Anything Nano (224x224) Results ---\n");
        printf("Output Map: 16x16\n");
        printf("Min Depth: %.4f | Max Depth: %.4f\n", min_depth, max_depth);
        printf("\nCenter 4x4 Output block:\n");
        for (int row = 6; row < 10; row++) {
            for (int col = 6; col < 10; col++) {
                printf("%6.2f ", out_floats[row * 16 + col]);
            }
            printf("\n");
        }
        printf("---------------------------------------------\n");
        
        printf("Writing output depth map to SD card (/sdcard/output.bin)...\n");
        FILE *f_out = fopen(MOUNT_POINT "/output.bin", "wb");
        if (f_out) {
            fwrite(out_floats, sizeof(float), output_elements, f_out);
            fclose(f_out);
            printf("Successfully saved output.bin\n");
        } else {
            printf("Failed to open output.bin for writing.\n");
        }
        
        free(out_floats);
    }
    
    free(img_buffer);
    delete model;
}

extern "C" void app_main(void)
{
    printf("Starting Depth Anything Nano Inference App\n");
    if (init_sdcard() == ESP_OK) {
        run_depth_anything();
    } else {
        printf("Bypassing inference due to SD card error.\n");
    }
}
