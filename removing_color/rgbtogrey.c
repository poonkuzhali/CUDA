#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s input_image.png output_image.png\n", argv[0]);
        return 1;
    }

    unsigned error;
    unsigned char* image = NULL;
    unsigned width, height;

    // Decode the input image
    error = lodepng_decode32_file(&image, &width, &height, argv[1]);
    if (error) {
        printf("Error decoding image: %s\n", lodepng_error_text(error));
        return 1;
    }

    // Convert to grayscale using the colorimetric method
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned char* pixel = &image[4 * (y * width + x)];

            // Calculate grayscale value using colorimetric weights
            unsigned char gray = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];

            // Set the RGB values to the calculated gray value
            pixel[0] = pixel[1] = pixel[2] = gray;
        }
    }

    // Encode the grayscale image
    error = lodepng_encode32_file(argv[2], image, width, height);
    if (error) {
        printf("Error encoding image: %s\n", lodepng_error_text(error));
        free(image);
        return 1;
    }

    free(image);
    printf("Image converted to grayscale successfully.\n");
    return 0;
}