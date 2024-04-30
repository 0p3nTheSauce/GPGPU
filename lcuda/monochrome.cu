#include <cuda_runtime.h>

// Define the pixel structure
struct pixel {
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;
};

// Define the test_params structure and set_up_test function
// (Assuming these are defined elsewhere in your codebase)

// Define KernelTimer and finish_test functions
// (Assuming these are defined elsewhere in your codebase)

// Kernel function to convert image to monochrome
__global__ void monochrome(const pixel *source, pixel *dest, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float value = source[index].red * 0.3125f + source[index].green * 0.5f + source[index].blue * .1875f;
    dest[index].red = static_cast<unsigned char>(value);
    dest[index].green = static_cast<unsigned char>(value);
    dest[index].blue = static_cast<unsigned char>(value);
    dest[index].alpha = source[index].alpha;
}

int main(int argc, char **argv)
{
    // Set up test parameters
    test_params params = set_up_test(argc, argv);

    int pixel_count = params.width * params.height;
    int BLOCK_SIZE = 128;
    int n_blocks = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    {
        // Measure kernel execution time
        KernelTimer t;
        // Launch monochrome kernel
        monochrome<<<n_blocks, BLOCK_SIZE>>>(params.input_image, params.output_image, pixel_count);
        // Check for kernel launch errors
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_error));
            return 1;
        }
    }

    // Finish test
    finish_test(params);

    return 0;
}
