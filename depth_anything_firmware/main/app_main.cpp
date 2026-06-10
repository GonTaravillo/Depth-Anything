#include "dl_model_base.hpp"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "esp_timer.h"
#include "esp_jpeg_enc.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <string>

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");
extern const uint8_t image0_bin_start[] asm("_binary_image0_bin_start");
extern const uint8_t image0_bin_end[] asm("_binary_image0_bin_end");
extern const uint8_t image1_bin_start[] asm("_binary_image1_bin_start");
extern const uint8_t image1_bin_end[] asm("_binary_image1_bin_end");
extern const uint8_t image2_bin_start[] asm("_binary_image2_bin_start");
extern const uint8_t image2_bin_end[] asm("_binary_image2_bin_end");
extern const uint8_t image3_bin_start[] asm("_binary_image3_bin_start");
extern const uint8_t image3_bin_end[] asm("_binary_image3_bin_end");
extern const uint8_t image4_bin_start[] asm("_binary_image4_bin_start");
extern const uint8_t image4_bin_end[] asm("_binary_image4_bin_end");

#define MOUNT_POINT "/sdcard"

// ESP32-S3-Korvo-2 microSD pin definitions (1-bit SDMMC mode)
#define SD_PIN_CLK  15
#define SD_PIN_CMD  7
#define SD_PIN_D0   4

static bool s_sd_mounted = false;

static esp_err_t init_sdcard(void)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };
    sdmmc_card_t *card;

    printf("Initializing SD card (CLK=%d, CMD=%d, D0=%d)...\n",
           SD_PIN_CLK, SD_PIN_CMD, SD_PIN_D0);

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1;          // 1-bit mode
    slot_config.clk   = (gpio_num_t)SD_PIN_CLK;
    slot_config.cmd   = (gpio_num_t)SD_PIN_CMD;
    slot_config.d0    = (gpio_num_t)SD_PIN_D0;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP; // enable internal pull-ups

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(MOUNT_POINT, &host, &slot_config, &mount_config, &card);
    if (ret != ESP_OK) {
        printf("SD mount failed: %s\n", esp_err_to_name(ret));
        return ret;
    }

    sdmmc_card_print_info(stdout, card);
    printf("SD card mounted at %s\n", MOUNT_POINT);
    s_sd_mounted = true;
    return ESP_OK;
}

// INFERNO colormap LUT
static const uint8_t inferno_lut[256][3] = {
    {0, 0, 3}, {0, 0, 4}, {0, 0, 6}, {1, 0, 7}, {1, 1, 9}, {1, 1, 11}, {2, 1, 14}, {2, 2, 16}, {3, 2, 18}, {4, 3, 20}, {4, 3, 22}, {5, 4, 24}, {6, 4, 27}, {7, 5, 29}, {8, 6, 31}, {9, 6, 33}, {10, 7, 35}, {11, 7, 38}, {13, 8, 40}, {14, 8, 42}, {15, 9, 45}, {16, 9, 47}, {18, 10, 50}, {19, 10, 52}, {20, 11, 54}, {22, 11, 57}, {23, 11, 59}, {25, 11, 62}, {26, 11, 64}, {28, 12, 67}, {29, 12, 69}, {31, 12, 71}, {32, 12, 74}, {34, 11, 76}, {36, 11, 78}, {38, 11, 80}, {39, 11, 82}, {41, 11, 84}, {43, 10, 86}, {45, 10, 88}, {46, 10, 90}, {48, 10, 92}, {50, 9, 93}, {52, 9, 95}, {53, 9, 96}, {55, 9, 97}, {57, 9, 98}, {59, 9, 100}, {60, 9, 101}, {62, 9, 102}, {64, 9, 102}, {65, 9, 103}, {67, 10, 104}, {69, 10, 105}, {70, 10, 105}, {72, 11, 106}, {74, 11, 106}, {75, 12, 107}, {77, 12, 107}, {79, 13, 108}, {80, 13, 108}, {82, 14, 108}, {83, 14, 109}, {85, 15, 109}, {87, 15, 109}, {88, 16, 109}, {90, 17, 109}, {91, 17, 110}, {93, 18, 110}, {95, 18, 110}, {96, 19, 110}, {98, 20, 110}, {99, 20, 110}, {101, 21, 110}, {102, 21, 110}, {104, 22, 110}, {106, 23, 110}, {107, 23, 110}, {109, 24, 110}, {110, 24, 110}, {112, 25, 110}, {114, 25, 109}, {115, 26, 109}, {117, 27, 109}, {118, 27, 109}, {120, 28, 109}, {122, 28, 109}, {123, 29, 108}, {125, 29, 108}, {126, 30, 108}, {128, 31, 107}, {129, 31, 107}, {131, 32, 107}, {133, 32, 106}, {134, 33, 106}, {136, 33, 106}, {137, 34, 105}, {139, 34, 105}, {141, 35, 105}, {142, 36, 104}, {144, 36, 104}, {145, 37, 103}, {147, 37, 103}, {149, 38, 102}, {150, 38, 102}, {152, 39, 101}, {153, 40, 100}, {155, 40, 100}, {156, 41, 99}, {158, 41, 99}, {160, 42, 98}, {161, 43, 97}, {163, 43, 97}, {164, 44, 96}, {166, 44, 95}, {167, 45, 95}, {169, 46, 94}, {171, 46, 93}, {172, 47, 92}, {174, 48, 91}, {175, 49, 91}, {177, 49, 90}, {178, 50, 89}, {180, 51, 88}, {181, 51, 87}, {183, 52, 86}, {184, 53, 86}, {186, 54, 85}, {187, 55, 84}, {189, 55, 83}, {190, 56, 82}, {191, 57, 81}, {193, 58, 80}, {194, 59, 79}, {196, 60, 78}, {197, 61, 77}, {199, 62, 76}, {200, 62, 75}, {201, 63, 74}, {203, 64, 73}, {204, 65, 72}, {205, 66, 71}, {207, 68, 70}, {208, 69, 68}, {209, 70, 67}, {210, 71, 66}, {212, 72, 65}, {213, 73, 64}, {214, 74, 63}, {215, 75, 62}, {217, 77, 61}, {218, 78, 59}, {219, 79, 58}, {220, 80, 57}, {221, 82, 56}, {222, 83, 55}, {223, 84, 54}, {224, 86, 52}, {226, 87, 51}, {227, 88, 50}, {228, 90, 49}, {229, 91, 48}, {230, 92, 46}, {230, 94, 45}, {231, 95, 44}, {232, 97, 43}, {233, 98, 42}, {234, 100, 40}, {235, 101, 39}, {236, 103, 38}, {237, 104, 37}, {237, 106, 35}, {238, 108, 34}, {239, 109, 33}, {240, 111, 31}, {240, 112, 30}, {241, 114, 29}, {242, 116, 28}, {242, 117, 26}, {243, 119, 25}, {243, 121, 24}, {244, 122, 22}, {245, 124, 21}, {245, 126, 20}, {246, 128, 18}, {246, 129, 17}, {247, 131, 16}, {247, 133, 14}, {248, 135, 13}, {248, 136, 12}, {248, 138, 11}, {249, 140, 9}, {249, 142, 8}, {249, 144, 8}, {250, 145, 7}, {250, 147, 6}, {250, 149, 6}, {250, 151, 6}, {251, 153, 6}, {251, 155, 6}, {251, 157, 6}, {251, 158, 7}, {251, 160, 7}, {251, 162, 8}, {251, 164, 10}, {251, 166, 11}, {251, 168, 13}, {251, 170, 14}, {251, 172, 16}, {251, 174, 18}, {251, 176, 20}, {251, 177, 22}, {251, 179, 24}, {251, 181, 26}, {251, 183, 28}, {251, 185, 30}, {250, 187, 33}, {250, 189, 35}, {250, 191, 37}, {250, 193, 40}, {249, 195, 42}, {249, 197, 44}, {249, 199, 47}, {248, 201, 49}, {248, 203, 52}, {248, 205, 55}, {247, 207, 58}, {247, 209, 60}, {246, 211, 63}, {246, 213, 66}, {245, 215, 69}, {245, 217, 72}, {244, 219, 75}, {244, 220, 79}, {243, 222, 82}, {243, 224, 86}, {243, 226, 89}, {242, 228, 93}, {242, 230, 96}, {241, 232, 100}, {241, 233, 104}, {241, 235, 108}, {241, 237, 112}, {241, 238, 116}, {241, 240, 121}, {241, 242, 125}, {242, 243, 129}, {242, 244, 133}, {243, 246, 137}, {244, 247, 141}, {245, 248, 145}, {246, 250, 149}, {247, 251, 153}, {249, 252, 157}, {250, 253, 160}, {252, 254, 164}
};

static void get_heatmap_color(float v, uint8_t &r, uint8_t &g, uint8_t &b)
{
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    int idx = (int)(v * 255.0f);
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    r = inferno_lut[idx][0];
    g = inferno_lut[idx][1];
    b = inferno_lut[idx][2];
}

void run_depth_anything(dl::Model *model, const uint8_t* img_start, const uint8_t* img_end, const char* out_filename)
{
    // -----------------------------------------------------------------------
    // 1. Get Model Inputs/Outputs
    // -----------------------------------------------------------------------

    std::map<std::string, dl::TensorBase *> model_inputs  = model->get_inputs();
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_input  = model_inputs.begin()->second;
    dl::TensorBase *model_output = model_outputs.begin()->second;

    // -----------------------------------------------------------------------
    // 2. Load input image from Flash
    // -----------------------------------------------------------------------
    const int width = 112, height = 112, channels = 3;
    const int num_pixels = width * height * channels;

    printf("Allocating input buffer (%d floats) in SPIRAM...\n", num_pixels);
    float *input_data = (float *)heap_caps_malloc(
        num_pixels * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!input_data) {
        return;
    }

    printf("Reading image from Flash...\n");
    const size_t expected_size = num_pixels * sizeof(float);
    const size_t image_size    = img_end - img_start;

    if (image_size != expected_size) {
        printf("ERROR: Flash image size mismatch. Expected %u bytes, got %u bytes.\n",
               (unsigned)expected_size, (unsigned)image_size);
        return;
    }
    memcpy(input_data, img_start, expected_size);
    printf("Copied %d floats from flash.\n", num_pixels);

    // -----------------------------------------------------------------------
    // 3. Quantize input and run inference
    // -----------------------------------------------------------------------
    printf("Quantizing input into model tensor...\n");
    int8_t *input_ptr = (int8_t *)model_input->data;
    int     exponent  = model_input->exponent;
    for (int i = 0; i < num_pixels; i++) {
        input_ptr[i] = dl::quantize<int8_t>(input_data[i], DL_RESCALE(exponent));
    }
    free(input_data); // no longer needed

    printf("Running inference...\n");
    long long t0 = esp_timer_get_time();
    model->run();
    long long t1 = esp_timer_get_time();
    printf("Inference completed in %lld ms\n", (t1 - t0) / 1000);

    // -----------------------------------------------------------------------
    // 4. Dequantize output
    // -----------------------------------------------------------------------
    printf("Processing output tensor...\n");
    int8_t *output_ptr = (int8_t *)model_output->data;
    int     out_exp    = model_output->exponent;

    std::vector<int> out_shape = model_output->get_shape();
    int output_elements = 1;
    for (int d : out_shape) output_elements *= d;

    printf("Output shape: ");
    for (int d : out_shape) printf("%d ", d);
    printf("\n");

    // Allocate output float buffer in SPIRAM
    float *out_floats = (float *)heap_caps_malloc(
        output_elements * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!out_floats) {
        return;
    }

    float min_depth =  1e9f;
    float max_depth = -1e9f;
    for (int i = 0; i < output_elements; i++) {
        float v = dl::dequantize(output_ptr[i], DL_SCALE(out_exp));
        out_floats[i] = v;
        if (v < min_depth) min_depth = v;
        if (v > max_depth) max_depth = v;
    }

    printf("\n--- Depth Anything Nano Results ---\n");
    printf("Min Depth: %.4f | Max Depth: %.4f\n", min_depth, max_depth);

    // Quick center-block preview
    const int out_h = out_shape[1], out_w = out_shape[2];
    printf("\nCenter 4x4 block:\n");
    for (int y = out_h / 2 - 2; y < out_h / 2 + 2; y++) {
        for (int x = out_w / 2 - 2; x < out_w / 2 + 2; x++) {
            if (y >= 0 && y < out_h && x >= 0 && x < out_w)
                printf("%6.2f ", out_floats[y * out_w + x]);
        }
        printf("\n");
    }

    // -----------------------------------------------------------------------
    // 5. Build RGB heatmap and encode as JPEG
    // -----------------------------------------------------------------------
    const float range = (max_depth - min_depth > 0.001f) ? (max_depth - min_depth) : 0.001f;
    const int   rgb_size = out_h * out_w * 3;

    printf("\nBuilding RGB heatmap...\n");
    uint8_t *rgb_buf = (uint8_t *)jpeg_calloc_align(rgb_size, 16);
    if (!rgb_buf) {
        printf("ERROR: Failed to allocate RGB buffer.\n");
        return;
    }

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float norm = (out_floats[y * out_w + x] - min_depth) / range;
            uint8_t r, g, b;
            get_heatmap_color(norm, r, g, b);
            int idx = (y * out_w + x) * 3;
            rgb_buf[idx]     = r;
            rgb_buf[idx + 1] = g;
            rgb_buf[idx + 2] = b;
        }
    }
    heap_caps_free(out_floats);

    printf("Encoding JPEG (quality=90)...\n");
    jpeg_enc_config_t enc_cfg = DEFAULT_JPEG_ENC_CONFIG();
    enc_cfg.width       = out_w;
    enc_cfg.height      = out_h;
    enc_cfg.src_type    = JPEG_PIXEL_FORMAT_RGB888;
    enc_cfg.subsampling = JPEG_SUBSAMPLE_444;
    enc_cfg.quality     = 90;

    jpeg_enc_handle_t jpeg_enc = NULL;
    jpeg_error_t      j_ret   = jpeg_enc_open(&enc_cfg, &jpeg_enc);

    if (j_ret != JPEG_ERR_OK || jpeg_enc == NULL) {
        printf("ERROR: Could not open JPEG encoder (%d).\n", j_ret);
        return;
    }

    uint8_t *jpg_buf  = (uint8_t *)jpeg_calloc_align(rgb_size, 16);
    int      jpg_size = 0;

    if (!jpg_buf) {
        printf("ERROR: Failed to allocate JPEG output buffer.\n");
    } else {
        j_ret = jpeg_enc_process(jpeg_enc, rgb_buf, rgb_size, jpg_buf, rgb_size, &jpg_size);
        if (j_ret == JPEG_ERR_OK) {
            printf("JPEG ready: %d bytes.\n", jpg_size);

            // ---------------------------------------------------------------
            // 6. Write to SD card
            // ---------------------------------------------------------------
            if (!s_sd_mounted) {
                printf("WARNING: SD card not mounted. Cannot save JPEG.\n");
            } else {
                FILE *f = fopen(out_filename, "wb");
                if (f == NULL) {
                    printf("ERROR: Could not open %s for writing.\n", out_filename);
                } else {
                    size_t written = fwrite(jpg_buf, 1, jpg_size, f);
                    fclose(f);
                    if ((int)written == jpg_size) {
                        printf("SUCCESS: Saved %u bytes to %s\n",
                               (unsigned)written, out_filename);
                    } else {
                        printf("ERROR: Only wrote %u / %d bytes.\n",
                               (unsigned)written, jpg_size);
                    }
                }
            }
        } else {
            printf("ERROR: JPEG encoding failed (%d).\n", j_ret);
        }
        jpeg_free_align(jpg_buf);
    }

    jpeg_enc_close(jpeg_enc);
    jpeg_free_align(rgb_buf);
}

extern "C" void app_main(void)
{
    printf("=== Depth Anything Nano – ESP32-S3-Korvo-2 ===\n");
    init_sdcard();   // failure is non-fatal: inference still runs
    printf("Loading model from Flash...\n");
    dl::Model *model = new dl::Model(
        (const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);

    const uint8_t* img_starts[5] = {image0_bin_start, image1_bin_start, image2_bin_start, image3_bin_start, image4_bin_start};
    const uint8_t* img_ends[5]   = {image0_bin_end, image1_bin_end, image2_bin_end, image3_bin_end, image4_bin_end};

    for (int i = 0; i < 5; i++) {
        char out_filename[64];
        sprintf(out_filename, "/sdcard/depth_out_%d.jpg", i);
        printf("\n=== Running Image %d ===\n", i);
        run_depth_anything(model, img_starts[i], img_ends[i], out_filename);
    }

    delete model;
    printf("=== Done ===\n");
}
