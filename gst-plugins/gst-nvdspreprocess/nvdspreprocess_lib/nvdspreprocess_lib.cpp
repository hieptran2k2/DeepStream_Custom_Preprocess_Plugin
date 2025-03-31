#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>   // cv::cuda::resize
#include <opencv2/cudaimgproc.hpp>    // cv::cuda::cvtColor

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "nvdspreprocess_lib.h"
#include "nvdspreprocess_impl.h"

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
  } \
} while (0)

struct CustomCtx
{
  /** Custom initialization parameters */
  CustomInitParams initParams;
  /** Custom mean subtraction and normalization parameters */
  CustomMeanSubandNormParams custom_mean_norm_params;
  /** unique pointer to tensor_impl class instance */
  std::unique_ptr <NvDsPreProcessTensorImpl> tensor_impl;
};

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
static gboolean
get_absolute_file_path (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[_PATH_MAX + 1];
  gchar abs_real_file_path[_PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else
      return FALSE;
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

NvDsPreProcessStatus
CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch, NvDsPreProcessCustomBuf *&buf,
                        CustomTensorParams &tensorParam, NvDsPreProcessAcquirer *acquirer)
{
  NvDsPreProcessStatus status = NVDSPREPROCESS_TENSOR_NOT_READY;

  /** acquire a buffer from tensor pool */
  buf = acquirer->acquire();

  /** Prepare Tensor */
  status = ctx->tensor_impl->prepare_tensor(batch, buf->memory_ptr);
  if (status != NVDSPREPROCESS_SUCCESS) {
    printf ("Custom Lib: Tensor Preparation failed\n");
    acquirer->release(buf);
  }

  /** synchronize cuda stream */
  status = ctx->tensor_impl->syncStream();
  if (status != NVDSPREPROCESS_SUCCESS) {
    printf ("Custom Lib: Cuda Stream Synchronization failed\n");
    acquirer->release(buf);
  }

  tensorParam.params.network_input_shape[0] = (int)batch->units.size();

  if (status != NVDSPREPROCESS_SUCCESS) {
    printf ("CustomTensorPreparation failed\n");
    acquirer->release(buf);
  }

  return status;
}

int MappingOpenCVInterpolation(NvBufSurfTransform_Inter filter)
{
  /*
   * @brief Maps a NvBufSurfTransform_Inter interpolation filter to the corresponding OpenCV interpolation method.
   *
   * @param filter A value of type NvBufSurfTransform_Inter that specifies the interpolation filter.
   * @return The corresponding OpenCV interpolation flag (e.g., cv::INTER_NEAREST, cv::INTER_LINEAR, etc.) if valid; 
   *         otherwise, returns -1 to indicate an error.
   */

  if (filter < OPEN_CV_INTER_NEAREST)
  {
      return -1;
  }

  int interpolation;

  switch (filter) {
      case OPEN_CV_INTER_NEAREST:
          interpolation = cv::INTER_NEAREST;
          break;
      case OPEN_CV_INTER_LINEAR:
          interpolation = cv::INTER_LINEAR;
          break;
      case OPEN_CV_INTER_CUBIC:
          interpolation = cv::INTER_CUBIC;
          break;
      case OPEN_CV_INTER_AREA:
          interpolation = cv::INTER_AREA;
          break;
      case OPEN_CV_INTER_LANCZOS4:
          interpolation = cv::INTER_LANCZOS4;
          break;
      case OPEN_CV_INTER_MAX:
          interpolation = cv::INTER_MAX;
          break;
      case OPEN_CV_WARP_FILL_OUTLIER:
          interpolation = cv::INTER_NEAREST; // Default for warp outlier
          break;
      case OPEN_CV_WARP_INVERSE_MAP:
          interpolation = cv::INTER_LINEAR; // Default for warp inverse map
          break;
      default:
          interpolation = -1;
          break;
  }

  return interpolation;
}

NvDsPreProcessStatus 
NvBufSurfaceConversion(NvBufSurface *in_surf, NvBufSurface *output_surf, CustomTransformParams &params)
{
  /*
   * @brief Converts the input NvBufSurface to a specified output surface with a desired color format using GPU-based transformation.
   *
   * This function configures the transformation session parameters, initializes the source and destination rectangles,
   * and performs the batched surface transformation from the input surface to the output surface.
   *
   * @param in_surf Pointer to the input NvBufSurface.
   * @param output_surf Pointer to the output NvBufSurface that will receive the transformed image.
   * @param params Reference to a CustomTransformParams object that holds the transformation configuration parameters.
   * @return NvDsPreProcessStatus status code indicating success (NVDSPREPROCESS_SUCCESS) or failure 
   *         (NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED) of the conversion.
   */

  NvBufSurfTransform_Error err;
  
  // Configure transformation session parameters
  NvBufSurfTransformConfigParams configParams;
  memset(&configParams, 0, sizeof(configParams));
  configParams.gpu_id = params.transform_config_params.gpu_id;
  configParams.cuda_stream = params.transform_config_params.cuda_stream;
  configParams.compute_mode = params.transform_config_params.compute_mode;

  err = NvBufSurfTransformSetSessionParams(&configParams);
  if (err != NvBufSurfTransformError_Success) {
    printf("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
    return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  NvBufSurfTransformRect* srcRect = new NvBufSurfTransformRect[in_surf->numFilled];
  NvBufSurfTransformRect* dstRect = new NvBufSurfTransformRect[in_surf->numFilled];

  for (guint frameIndex = 0; frameIndex < in_surf->numFilled; frameIndex++) {
    // Initialize the source rectangle: using the entire frame
    srcRect[frameIndex].left   = 0;
    srcRect[frameIndex].top    = 0;
    srcRect[frameIndex].width  = in_surf->surfaceList[frameIndex].width;
    srcRect[frameIndex].height = in_surf->surfaceList[frameIndex].height;
    
    // Initialize the destination rectangle (currently the same size; can be adjusted if needed)
    dstRect[frameIndex].left   = 0;
    dstRect[frameIndex].top    = 0;
    dstRect[frameIndex].width  = in_surf->surfaceList[frameIndex].width;
    dstRect[frameIndex].height = in_surf->surfaceList[frameIndex].height;
  }

  // Set transformation parameters
  NvBufSurfTransformParams transformParams;
  memset(&transformParams, 0, sizeof(transformParams));
  transformParams.src_rect = srcRect;
  transformParams.dst_rect = dstRect;
  transformParams.transform_flip = NvBufSurfTransform_None;
  transformParams.transform_flag = NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST;
  transformParams.transform_filter = NvBufSurfTransformInter_Default;

  // Perform the surface transformation
  err = NvBufSurfTransform(in_surf, output_surf, &transformParams);
  if (err != NvBufSurfTransformError_Success)
  {
      printf("NvBufSurfTransform failed with error %d\n", err);
      delete[] srcRect;
      delete[] dstRect;
      return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  delete[] srcRect;
  delete[] dstRect;
  return NVDSPREPROCESS_SUCCESS;
}

// Function to perform an OpenCV transformation on the input surface and write the result to the output surface
NvDsPreProcessStatus
OpenCVTransform_CPU(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
  /*
   * @brief Performs an OpenCV-based transformation on the input surface.
   *
   * This function converts the input surface to an RGBA format on the GPU, transfers the image data to host memory,
   * applies cropping and resizing operations using OpenCV, and then copies the transformed image back to the GPU output surface.
   *
   * @param in_surf Pointer to the input NvBufSurface.
   * @param out_surf Pointer to the output NvBufSurface where the transformed image will be stored.
   * @param params Reference to a CustomTransformParams object containing transformation parameters (such as cropping and resizing settings).
   * @return NvDsPreProcessStatus status code indicating success (NVDSPREPROCESS_SUCCESS) or failure 
   *         (NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED) of the transformation.
   */

  NvDsPreProcessStatus err;

  NvBufSurfTransform_Inter filter = params.transform_params.transform_filter;

  // Tạo surface mới trên GPU để lưu kết quả chuyển đổi sang RGB
  NvBufSurfaceCreateParams output_param;
  memset(&output_param, 0, sizeof(output_param));
  output_param.gpuId = in_surf->gpuId;
  output_param.colorFormat = NVBUF_COLOR_FORMAT_RGBA;   // Desired color format (e.g., convert NV12 to RGBA)
  output_param.memType = NVBUF_MEM_DEFAULT;  // Memory type (e.g., CUDA memory)
  output_param.layout = NVBUF_LAYOUT_PITCH;;
  // output_param.isContiguous = in_surf->isContiguous;
  output_param.width = in_surf->surfaceList[0].width;  // Surface width
  output_param.height = in_surf->surfaceList[0].height; // Surface height

  NvBufSurface* in_RGBA_surf = nullptr;
  if (NvBufSurfaceCreate(&in_RGBA_surf, in_surf->numFilled, &output_param) != 0) {
    std::cerr << "Error: Không thể cấp phát buffer đích." << std::endl;
    return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  // Convert the input surface to RGBA
  err = NvBufSurfaceConversion(in_surf, in_RGBA_surf, params);
  if (err != NVDSPREPROCESS_SUCCESS)
  {
    NvBufSurfaceUnMap(in_RGBA_surf, 0, 0);
    NvBufSurfaceDestroy(in_RGBA_surf);
    printf("NvBufSurfTransform failed with error %d\n", err);
    return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  // Process each filled frame in the surface
  for (guint frameIndex = 0; frameIndex < in_surf->numFilled; frameIndex++) {

    // Retrieve frame dimensions
    gint frame_width = (gint)in_RGBA_surf->surfaceList[frameIndex].width;
    gint frame_height = (gint)in_RGBA_surf->surfaceList[frameIndex].height;

    // Allocate host memory for copying image data from the GPU
    void *src_data = NULL;
    CHECK_CUDA_STATUS (cudaMallocHost (&src_data,
                                       in_RGBA_surf->surfaceList[frameIndex].dataSize), "Could not allocate cuda host buffer");
    
    if (src_data == NULL) {
      g_print("Error: failed to malloc src_data \n");
      return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    // Copy image data from device (GPU) to host (CPU)
    cudaMemcpy((void *)src_data,
        (void *)in_RGBA_surf->surfaceList[frameIndex].dataPtr,
        in_RGBA_surf->surfaceList[frameIndex].dataSize,
        cudaMemcpyDeviceToHost);

    size_t frame_step = in_RGBA_surf->surfaceList[frameIndex].pitch;

    // Create an OpenCV Mat from the GPU-copied image data
    cv::Mat frame_mat = cv::Mat(frame_height, frame_width, CV_8UC4, src_data, frame_step);

    // Define the region of interest for cropping based on source rectangle parameters
    cv::Rect roi(params.transform_params.src_rect[frameIndex].left, params.transform_params.src_rect[frameIndex].top,  
                 params.transform_params.src_rect[frameIndex].width, params.transform_params.src_rect[frameIndex].height);

    cv::Mat cropped_image = frame_mat(roi);

    // Map the interpolation filter to the corresponding OpenCV interpolation method
    int interpolation = MappingOpenCVInterpolation(filter);
    if (interpolation == -1)
    {
        printf("Error: Invalid interpolation filter mapping.\n");
        cudaFreeHost(src_data);
        NvBufSurfaceUnMap(in_RGBA_surf, 0, 0);
        NvBufSurfaceDestroy(in_RGBA_surf);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    // Resize the cropped image to match the destination rectangle dimensions
    cv::Size newSize(params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height);
    cv::Mat resize_img;
    cv::resize(cropped_image, resize_img, newSize, 0, 0, interpolation); 

    // Prepare the output image
    cv::Mat out_image;
    if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGBA)
    {
      // Create a black RGBA image of the required output dimensions
      out_image = cv::Mat::zeros(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC4);

      // Copy the resized image into the designated destination region
      resize_img.copyTo(out_image(cv::Rect(params.transform_params.dst_rect[frameIndex].left, params.transform_params.dst_rect[frameIndex].top,  
                                          params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height)));                                                         
    } else if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGB) {
          cv::Mat rgb_gpu;
          cv::cuda::cvtColor(resize_gpu, rgb_gpu, cv::COLOR_RGBA2RGB);
          out_image = cv::Mat::zeros(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC3); 
          rgb_gpu.copyTo(out_image(cv::Rect(params.transform_params.dst_rect[frameIndex].left, params.transform_params.dst_rect[frameIndex].top,  
                                          params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height)));
    } else {
      // Create a black grayscale image if the output format is GRAY
      out_image = cv::Mat::zeros(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC1); 

      cv::Mat gray_image;
      cv::cvtColor(resize_img, gray_image, cv::COLOR_RGBA2GRAY);
      gray_image.copyTo(out_image(cv::Rect(params.transform_params.dst_rect[frameIndex].left, params.transform_params.dst_rect[frameIndex].top,  
                                          params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height)));   
    }

    // Copy the processed output image from host memory back to the GPU
    size_t sizeInBytes = out_surf->surfaceList[frameIndex].dataSize;
    cudaMemcpy((void *)out_surf->surfaceList[frameIndex].dataPtr,
        out_image.ptr(0),
        sizeInBytes,
        cudaMemcpyHostToDevice);  

    // Free the allocated host memory for this frame
    cudaFreeHost(src_data);
  }

  // Unmap and destroy the intermediate RGBA surface
  NvBufSurfaceUnMap(in_RGBA_surf, 0, 0);
  NvBufSurfaceDestroy(in_RGBA_surf);

  return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
OpenCVTransform_CPU_Async(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
    NvDsPreProcessStatus err;
    NvBufSurfTransform_Inter filter = params.transform_params.transform_filter;

    // Create a CUDA stream for asynchronous memory copies
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "Error: Unable to create CUDA stream." << std::endl;
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    // Create a new GPU surface to store the converted RGBA image
    NvBufSurfaceCreateParams output_param;
    memset(&output_param, 0, sizeof(output_param));
    output_param.gpuId = in_surf->gpuId;
    output_param.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    output_param.memType = NVBUF_MEM_DEFAULT;
    output_param.layout = NVBUF_LAYOUT_PITCH;
    output_param.width = in_surf->surfaceList[0].width;
    output_param.height = in_surf->surfaceList[0].height;

    NvBufSurface* in_RGBA_surf = nullptr;
    if (NvBufSurfaceCreate(&in_RGBA_surf, in_surf->numFilled, &output_param) != 0) {
        std::cerr << "Error: Unable to allocate destination buffer." << std::endl;
        cudaStreamDestroy(stream);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    // Convert the input surface to RGBA on GPU
    err = NvBufSurfaceConversion(in_surf, in_RGBA_surf, params);
    if (err != NVDSPREPROCESS_SUCCESS) {
        NvBufSurfaceUnMap(in_RGBA_surf, 0, 0);
        NvBufSurfaceDestroy(in_RGBA_surf);
        cudaStreamDestroy(stream);
        printf("NvBufSurfTransform failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    // Process each filled frame in the surface
    for (guint frameIndex = 0; frameIndex < in_surf->numFilled; frameIndex++) {

        // Retrieve frame dimensions
        gint frame_width = in_RGBA_surf->surfaceList[frameIndex].width;
        gint frame_height = in_RGBA_surf->surfaceList[frameIndex].height;

        // Allocate host pinned memory for copying image data from the GPU
        void *src_data = NULL;
        CHECK_CUDA_STATUS(cudaMallocHost(&src_data, in_RGBA_surf->surfaceList[frameIndex].dataSize),
                          "Could not allocate cuda host buffer");
        if (src_data == NULL) {
            g_print("Error: failed to malloc src_data \n");
            cudaStreamDestroy(stream);
            return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
        }

        // Copy image data from device (GPU) to host (CPU) asynchronously
        cudaMemcpyAsync(src_data,
                        in_RGBA_surf->surfaceList[frameIndex].dataPtr,
                        in_RGBA_surf->surfaceList[frameIndex].dataSize,
                        cudaMemcpyDeviceToHost, stream);

        // Synchronize the stream to ensure the copy is complete before CPU processing
        cudaStreamSynchronize(stream);

        size_t frame_step = in_RGBA_surf->surfaceList[frameIndex].pitch;

        // Create an OpenCV Mat from the host-copied image data
        cv::Mat frame_mat(frame_height, frame_width, CV_8UC4, src_data, frame_step);

        // Define the region of interest for cropping
        cv::Rect roi(params.transform_params.src_rect[frameIndex].left, params.transform_params.src_rect[frameIndex].top,
                     params.transform_params.src_rect[frameIndex].width, params.transform_params.src_rect[frameIndex].height);
        cv::Mat cropped_image = frame_mat(roi);

        // Map the interpolation filter to the corresponding OpenCV interpolation method
        int interpolation = MappingOpenCVInterpolation(filter);
        if (interpolation == -1) {
            printf("Error: Invalid interpolation filter mapping.\n");
            cudaFreeHost(src_data);
            NvBufSurfaceUnMap(in_RGBA_surf, 0, 0);
            NvBufSurfaceDestroy(in_RGBA_surf);
            cudaStreamDestroy(stream);
            return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
        }

        // Resize the cropped image to match the destination rectangle dimensions
        cv::Size newSize(params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height);
        cv::Mat resize_img;
        cv::resize(cropped_image, resize_img, newSize, 0, 0, interpolation);

        // Prepare the output image
        cv::Mat out_image;
        if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
            // Create a black RGBA image of the required output dimensions
            out_image = cv::Mat::zeros(out_surf->surfaceList[frameIndex].height,
                                       out_surf->surfaceList[frameIndex].width, CV_8UC4);
            // Copy the resized image into the designated destination region
            resize_img.copyTo(out_image(cv::Rect(params.transform_params.dst_rect[frameIndex].left,
                                                 params.transform_params.dst_rect[frameIndex].top,
                                                 params.transform_params.dst_rect[frameIndex].width,
                                                 params.transform_params.dst_rect[frameIndex].height)));
        } else if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGB) {
          cv::Mat rgb_gpu;
          cv::cuda::cvtColor(resize_gpu, rgb_gpu, cv::COLOR_RGBA2RGB);
          out_image = cv::Mat::zeros(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC3); 
          rgb_gpu.copyTo(out_image(cv::Rect(params.transform_params.dst_rect[frameIndex].left, 
                                            params.transform_params.dst_rect[frameIndex].top,  
                                            params.transform_params.dst_rect[frameIndex].width, 
                                            params.transform_params.dst_rect[frameIndex].height)));
        } else {
            // Create a black grayscale image if the output format is GRAY
            out_image = cv::Mat::zeros(out_surf->surfaceList[frameIndex].height,
                                       out_surf->surfaceList[frameIndex].width, CV_8UC1);
            cv::Mat gray_image;
            cv::cvtColor(resize_img, gray_image, cv::COLOR_RGBA2GRAY);
            gray_image.copyTo(out_image(cv::Rect(params.transform_params.dst_rect[frameIndex].left,
                                                 params.transform_params.dst_rect[frameIndex].top,
                                                 params.transform_params.dst_rect[frameIndex].width,
                                                 params.transform_params.dst_rect[frameIndex].height)));
        }

        // Copy the processed output image from host memory back to the GPU asynchronously
        size_t sizeInBytes = out_surf->surfaceList[frameIndex].dataSize;
        cudaMemcpyAsync(out_surf->surfaceList[frameIndex].dataPtr,
                        out_image.ptr(0),
                        sizeInBytes,
                        cudaMemcpyHostToDevice, stream);

        // Synchronize the stream to ensure copy completes before moving to the next frame
        cudaStreamSynchronize(stream);

        // Free the allocated host memory for this frame
        cudaFreeHost(src_data);
    }

    // Unmap and destroy the intermediate RGBA surface
    NvBufSurfaceUnMap(in_RGBA_surf, 0, 0);
    NvBufSurfaceDestroy(in_RGBA_surf);

    // Destroy the CUDA stream
    cudaStreamDestroy(stream);

    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus 
OpenCVTransform_GPU(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
    NvDsPreProcessStatus err;
    NvBufSurfTransform_Inter filter = params.transform_params.transform_filter;

    // Convert input surface to RGBA format on GPU
    NvBufSurfaceCreateParams output_param = {};
    output_param.gpuId = in_surf->gpuId;
    output_param.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    output_param.memType = NVBUF_MEM_DEFAULT;
    output_param.layout = NVBUF_LAYOUT_PITCH;
    output_param.width = in_surf->surfaceList[0].width;
    output_param.height = in_surf->surfaceList[0].height;

    NvBufSurface* in_RGBA_surf = nullptr;
    if (NvBufSurfaceCreate(&in_RGBA_surf, in_surf->numFilled, &output_param) != 0) {
        std::cerr << "Error: Unable to allocate destination buffer." << std::endl;
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    err = NvBufSurfaceConversion(in_surf, in_RGBA_surf, params);
    if (err != NVDSPREPROCESS_SUCCESS) {
        NvBufSurfaceDestroy(in_RGBA_surf);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    cudaSetDevice(in_surf->gpuId);
    cv::cuda::setDevice(in_surf->gpuId);

    for (guint frameIndex = 0; frameIndex < in_surf->numFilled; frameIndex++) {
        int frame_width = in_RGBA_surf->surfaceList[frameIndex].width;
        int frame_height = in_RGBA_surf->surfaceList[frameIndex].height;
        
        // Create GpuMat from GPU data
        cv::cuda::GpuMat gpu_frame(frame_height, frame_width, CV_8UC4, in_RGBA_surf->surfaceList[frameIndex].dataPtr, in_RGBA_surf->surfaceList[frameIndex].pitch);

        // Crop image using ROI
        cv::Rect roi(params.transform_params.src_rect[frameIndex].left, params.transform_params.src_rect[frameIndex].top, params.transform_params.src_rect[frameIndex].width, params.transform_params.src_rect[frameIndex].height);
        cv::cuda::GpuMat cropped_gpu = gpu_frame(roi);
        
        // Resize image on GPU
        int interpolation = MappingOpenCVInterpolation(filter);
        if (interpolation == -1) {
            NvBufSurfaceDestroy(in_RGBA_surf);
            return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
        }
        cv::cuda::GpuMat resized_gpu;
        cv::cuda::resize(cropped_gpu, resized_gpu, cv::Size(params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height), 0, 0, interpolation);
        
        // Process output format
        cv::cuda::GpuMat output_gpu;
        if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
            output_gpu = cv::cuda::GpuMat(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC4);
            output_gpu.setTo(cv::Scalar(0, 0, 0, 0)); // Black image
            resized_gpu.copyTo(output_gpu(cv::Rect(params.transform_params.dst_rect[frameIndex].left, params.transform_params.dst_rect[frameIndex].top, params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height)));
        } else if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGB) {
          cv::cuda::GpuMat rgb_gpu;
          cv::cuda::cvtColor(resized_gpu, rgb_gpu, cv::COLOR_RGBA2RGB);
          output_gpu = cv::cuda::GpuMat(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC3);
          output_gpu.setTo(cv::Scalar(0, 0, 0)); // Black image
          rgb_gpu.copyTo(output_gpu(cv::Rect(nvinferpreprocess->transform_params.dst_rect[frameIndex].left, nvinferpreprocess->transform_params.dst_rect[frameIndex].top, nvinferpreprocess->transform_params.dst_rect[frameIndex].width, nvinferpreprocess->transform_params.dst_rect[frameIndex].height)));        
        } else {
            cv::cuda::GpuMat gray_gpu;
            cv::cuda::cvtColor(resized_gpu, gray_gpu, cv::COLOR_RGBA2GRAY);
            output_gpu = cv::cuda::GpuMat(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC1);
            output_gpu.setTo(cv::Scalar(0));
            gray_gpu.copyTo(output_gpu(cv::Rect(params.transform_params.dst_rect[frameIndex].left, params.transform_params.dst_rect[frameIndex].top, params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height)));
        }

        // Copy processed output image from GPU to output surface
        cudaMemcpy(out_surf->surfaceList[frameIndex].dataPtr, output_gpu.ptr(0), out_surf->surfaceList[frameIndex].dataSize, cudaMemcpyDeviceToDevice);
    }
    
    NvBufSurfaceDestroy(in_RGBA_surf);
    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus OpenCVTransform_GPU_Async(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
    NvDsPreProcessStatus err;
    NvBufSurfTransform_Inter filter = params.transform_params.transform_filter;

    // Create CUDA stream for asynchronous operations
    cv::cuda::Stream stream;

    // Convert input surface to RGBA format on GPU
    NvBufSurfaceCreateParams output_param = {};
    output_param.gpuId = in_surf->gpuId;
    output_param.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    output_param.memType = NVBUF_MEM_DEFAULT;
    output_param.layout = NVBUF_LAYOUT_PITCH;
    output_param.width = in_surf->surfaceList[0].width;
    output_param.height = in_surf->surfaceList[0].height;

    NvBufSurface* in_RGBA_surf = nullptr;
    if (NvBufSurfaceCreate(&in_RGBA_surf, in_surf->numFilled, &output_param) != 0) {
        std::cerr << "Error: Unable to allocate destination buffer." << std::endl;
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    err = NvBufSurfaceConversion(in_surf, in_RGBA_surf, params);
    if (err != NVDSPREPROCESS_SUCCESS) {
        NvBufSurfaceDestroy(in_RGBA_surf);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    cudaSetDevice(in_surf->gpuId);
    cv::cuda::setDevice(in_surf->gpuId);

    for (guint frameIndex = 0; frameIndex < in_surf->numFilled; frameIndex++) {
        int frame_width = in_RGBA_surf->surfaceList[frameIndex].width;
        int frame_height = in_RGBA_surf->surfaceList[frameIndex].height;
        
        // Create GpuMat from GPU data
        cv::cuda::GpuMat gpu_frame(frame_height, frame_width, CV_8UC4, in_RGBA_surf->surfaceList[frameIndex].dataPtr, in_RGBA_surf->surfaceList[frameIndex].pitch);
        
        // Crop image using ROI
        cv::Rect roi(params.transform_params.src_rect[frameIndex].left, params.transform_params.src_rect[frameIndex].top, 
                     params.transform_params.src_rect[frameIndex].width, params.transform_params.src_rect[frameIndex].height);
        cv::cuda::GpuMat cropped_gpu = gpu_frame(roi);
        
        // Resize image on GPU asynchronously using stream
        int interpolation = MappingOpenCVInterpolation(filter);
        if (interpolation == -1) {
            NvBufSurfaceDestroy(in_RGBA_surf);
            return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
        }
        cv::cuda::GpuMat resized_gpu;
        cv::cuda::resize(cropped_gpu, resized_gpu, cv::Size(params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height), 0, 0, interpolation, stream);
        
        // Process output format asynchronously
        cv::cuda::GpuMat output_gpu;
        if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
            output_gpu = cv::cuda::GpuMat(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC4);
            output_gpu.setTo(cv::Scalar(0, 0, 0, 0), stream); // Black image, set using stream
            resized_gpu.copyTo(output_gpu(cv::Rect(params.transform_params.dst_rect[frameIndex].left, params.transform_params.dst_rect[frameIndex].top, 
                                                   params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height)), stream);
        } else if (out_surf->surfaceList[frameIndex].colorFormat == NVBUF_COLOR_FORMAT_RGB) {
          cv::cuda::GpuMat rgb_gpu;
          cv::cuda::cvtColor(resized_gpu, rgb_gpu, cv::COLOR_RGBA2RGB);
          output_gpu = cv::cuda::GpuMat(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC3);
          output_gpu.setTo(cv::Scalar(0, 0, 0)); // Black image
          rgb_gpu.copyTo(output_gpu(cv::Rect(nvinferpreprocess->transform_params.dst_rect[frameIndex].left, nvinferpreprocess->transform_params.dst_rect[frameIndex].top, nvinferpreprocess->transform_params.dst_rect[frameIndex].width, nvinferpreprocess->transform_params.dst_rect[frameIndex].height)));
        } else {
            cv::cuda::GpuMat gray_gpu;
            cv::cuda::cvtColor(resized_gpu, gray_gpu, cv::COLOR_RGBA2GRAY, 0, stream);
            output_gpu = cv::cuda::GpuMat(out_surf->surfaceList[frameIndex].height, out_surf->surfaceList[frameIndex].width, CV_8UC1);
            output_gpu.setTo(cv::Scalar(0), stream);
            gray_gpu.copyTo(output_gpu(cv::Rect(params.transform_params.dst_rect[frameIndex].left, params.transform_params.dst_rect[frameIndex].top, 
                                                params.transform_params.dst_rect[frameIndex].width, params.transform_params.dst_rect[frameIndex].height)), stream);
        }
        
        // Use asynchronous memory copy
        cudaMemcpyAsync(out_surf->surfaceList[frameIndex].dataPtr,
          output_gpu.ptr(0),
          out_surf->surfaceList[frameIndex].dataSize,
          cudaMemcpyDeviceToDevice,
          reinterpret_cast<cudaStream_t>(stream.cudaPtr()));
    }
    
    // Wait for all asynchronous operations to complete before proceeding
    stream.waitForCompletion();
    
    NvBufSurfaceDestroy(in_RGBA_surf);
    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
CustomTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
  /*
   * @brief Applies a custom transformation on the input surface using either OpenCV-based processing or a batched GPU transformation.
   *
   * This function determines the transformation method based on the interpolation filter value. If the filter value is greater than
   * NvBufSurfTransformInter_Default, it applies an OpenCV-based transformation; otherwise, it performs a batched GPU transformation.
   *
   * @param in_surf Pointer to the input NvBufSurface.
   * @param out_surf Pointer to the output NvBufSurface where the transformed image will be stored.
   * @param params Reference to a CustomTransformParams object that holds both transformation configuration and parameters.
   * @return NvDsPreProcessStatus status code indicating success (NVDSPREPROCESS_SUCCESS) or failure 
   *         (NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED) of the custom transformation.
   */

  NvBufSurfTransform_Inter filter = params.transform_params.transform_filter;
  if (filter > NvBufSurfTransformInter_Default)
  {
    NvDsPreProcessStatus err_CV;
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
      err_CV = OpenCVTransform_CPU(in_surf, out_surf, params);
    } else {
      err_CV = OpenCVTransform_GPU(in_surf, out_surf, params);
    }
    if (err_CV != NVDSPREPROCESS_SUCCESS)
    {
        printf("OpenCVTransform failed with error %d\n", err_CV);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }
  }
  else
  {
    NvBufSurfTransform_Error err;
    err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success)
    {
        printf("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }
  
    /* Batched tranformation. */
    err = NvBufSurfTransform(in_surf, out_surf, &params.transform_params);
  
    if (err != NvBufSurfTransformError_Success)
    {
        printf("NvBufSurfTransform failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }
  }

  return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
CustomAsyncTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
  NvBufSurfTransform_Inter filter = params.transform_params.transform_filter;
  if (filter > NvBufSurfTransformInter_Default)
  {
    NvDsPreProcessStatus err_CV;
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
      err_CV = OpenCVTransform_CPU_Async(in_surf, out_surf, params);
    } else {
      err_CV = OpenCVTransform_GPU_Async(in_surf, out_surf, params);
    }
    if (err_CV != NVDSPREPROCESS_SUCCESS)
    {
        printf("OpenCVTransform failed with error %d\n", err_CV);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }
  }
  else
  {
    NvBufSurfTransform_Error err;
    err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success)
    {
        printf("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    /* Async Batched tranformation. */
    err = NvBufSurfTransformAsync(in_surf, out_surf, &params.transform_params, &params.sync_obj);

    if (err != NvBufSurfTransformError_Success)
    {
        printf("NvBufSurfTransform failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }
  }

  return NVDSPREPROCESS_SUCCESS;
}

CustomCtx *initLib(CustomInitParams initparams)
{
  auto ctx = std::make_unique<CustomCtx>();
  NvDsPreProcessStatus status;

  ctx->custom_mean_norm_params.pixel_normalization_factor =
      std::stof(initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_PIXEL_NORMALIZATION_FACTOR]);

  if (!initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].empty()) {
    char abs_path[_PATH_MAX] = {0};
    if (!get_absolute_file_path (initparams.config_file_path,
          initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].c_str(), abs_path)) {
      printf("Error: Could not parse mean image file path\n");
      return nullptr;
    }
    if (!ctx->custom_mean_norm_params.meanImageFilePath.empty()) {
      ctx->custom_mean_norm_params.meanImageFilePath.clear();
    }
    ctx->custom_mean_norm_params.meanImageFilePath.append(abs_path);
  }

  std::string offsets_str = initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_OFFSETS];

  if (!offsets_str.empty()) {
    std::string delimiter = ";";
    size_t pos = 0;
    std::string token;

    while ((pos = offsets_str.find(delimiter)) != std::string::npos) {
        token = offsets_str.substr(0, pos);
        ctx->custom_mean_norm_params.offsets.push_back(std::stof(token));
        offsets_str.erase(0, pos + delimiter.length());
    }
    ctx->custom_mean_norm_params.offsets.push_back(std::stof(offsets_str));

    printf("Using offsets : %f,%f,%f\n", ctx->custom_mean_norm_params.offsets[0],
          ctx->custom_mean_norm_params.offsets[1], ctx->custom_mean_norm_params.offsets[2]);
  }

  status = normalization_mean_subtraction_impl_initialize(&ctx->custom_mean_norm_params,
          &initparams.tensor_params, ctx->tensor_impl, initparams.unique_id);

  if (status != NVDSPREPROCESS_SUCCESS) {
    printf("normalization_mean_subtraction_impl_initialize failed\n");
    return nullptr;
  }

  ctx->initParams = initparams;

  return ctx.release();
}

void deInitLib(CustomCtx *ctx)
{
  delete ctx;
}
