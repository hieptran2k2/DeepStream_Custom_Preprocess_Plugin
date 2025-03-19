# DeepStream_Custom_Preprocess_Plugin
Create a custom preprocessing plugin in DeepStream using OpenCV with a scaling filter.
--------------------------------------------------------------------------------------------------
### Improvements on this repository

* Custom preprocess plugin with openCV interpolation
* Resolve the scaling issue of OpenCV and NVBUF, which differs according to the [forums](https://forums.developer.nvidia.com/t/image-comparison-deepstream-vs-opencv-python/220244)

### Basic usage

#### 1. Download the repo

```
git clone https://github.com/hieptran2k2/DeepStream_Custom_Preprocess_Plugin.git
cd DeepStream_Custom_Preprocess_Plugin
```

#### 3. Compile the lib

3.1. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
export NVDS_VERSION=X.Y (example: DeepStream 7.1, then NVDS_VERSION=7.1)
```

* x86 platform

  ```
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  DeepStream 5.1 = 11.1
  ```

* Jetson platform

  ```
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 / 5.1 = 10.2
  ```
#### 3. Install open-cv lib

3.1 OpenCV 
```
apt update
apt install build-essential cmake git pkg-config
apt install libjpeg-dev libpng-dev libtiff-dev
apt-get install -y libopencv-dev
```
Alternatively, you can follow the official OpenCV installation guide for Linux from [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

3.2 OpenCV Cuda
```
bash install_opencv_cuda.sh
```

#### 4. Optional Settings

4.1 Save Image Input Model

Navigate to the ```nvdsinfer``` directory:
```
cd DeepStream_Custom_Preprocess_Plugin/libs/nvdsinfer
```
To enable output saving, add the flag ```DUMP_INPUT_TO_FILE``` in the ```Makefile```:
```
...
CFLAGS+= -fPIC -Wno-deprecated-declarations -std=c++14 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes -DNDEBUG -DDUMP_INPUT_TO_FILE
...
```
Otherwise, use the following configuration:
```
...
CFLAGS+= -fPIC -Wno-deprecated-declarations -std=c++14 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes
...
```
To configure the output folder in the nvdsinfer, edit the ```nvdsinfer_context_impl.cpp```:
```
#define SAVE_FOLDER   "/workspace/output/" # folder save output in preprocess plugin
#define DUMP_FRAME_CNT_START   0           # start frame save
#define DUMP_FRAME_CNT_STOP    1000        # end frame save
#define dumpToRaw   false                  # save raw file or image
```

4.2 Save Output from Preprocess Plugin

Navigate to the ```gst-nvdspreprocess``` directory
```
cd DeepStream_Custom_Preprocess_Plugin/gst-plugins/gst-nvdspreprocess
```
To enable output saving, add the flag ```DUMP_ROIS``` in the ```Makefile```:
```
...
CFLAGS+= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror -DDS_VERSION=\"6.3.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I include \
	 -I ../../includes -DDUMP_ROIS
...
```
Otherwise, use the following configuration:
```
...
CFLAGS+= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror -DDS_VERSION=\"6.3.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I include \
	 -I ../../includes
...
```
To configure the output folder in the preprocess plugin, edit the ```gstnvdspreprocess.cpp``` file:
```
#define SAVE_IN_FOLDER  "/workspace/output/" # folder save output in preprocess plugin
```

#### 5. Build the Library

Run the following command to set up the environment and compile the library:
```
bash setting_enviroment.sh path/to/folder/DeepStream_Custom_Preprocess_Plugin
```

#### 6. Configure the Scaling Filter

Edit the ```config_preprocess.txt``` file to specify the scaling filter you wish to use:
```
[property]
...
    # Scaling interpolation method
    # 0 = NvBufSurfTransformInter_Nearest 
    # 1 = NvBufSurfTransformInter_Bilinear 
    # 2 = NvBufSurfTransformInter_Algo1
    # 3 = NvBufSurfTransformInter_Algo2 
    # 4 = NvBufSurfTransformInter_Algo3 
    # 5 = NvBufSurfTransformInter_Algo4
    # 6 = NvBufSurfTransformInter_Default
    # 7 = OPEN_CV_INTER_NEAREST 
    # 8 = OPEN_CV_INTER_LINEAR
    # 9 = OPEN_CV_INTER_CUBIC 
    # 10 = OPEN_CV_INTER_AREA 
    # 11 = OPEN_CV_INTER_LANCZOS4
    # 12 = OPEN_CV_INTER_MAX 
    # 13 = OPEN_CV_WARP_FILL_OUTLIER 
    # 14 = OPEN_CV_WARP_INVERSE_MAP
scaling-filter=8
...
[group-0]
src-ids=0;1
custom-input-transformation-function=CustomTransformation
....
```
#### 7. Run the Application

Run the application with the following command:
```
python main.py -i <uri1> [uri2] -o /path/to/output/file -c /path/to/folder/config
```
* Note

|       Flag          |                                   Describe                             |                             Example                          |
| :-----------------: | :--------------------------------------------------------------------: | :----------------------------------------------------------: |
| -i or --input       |      Path to input streams                                             | file:///path/to/file (h264, mp4, ...)  or rtsp://host/video1 |
| -o or --output      |      Path to output file                                               |                          /output/out.mp4                     |
| -c or  --configfile |      Choose the config-file to be used with specified pgie             |                      /model/pgie/config.txt                  |
| --file-loop         |      Loop the input file sources after EOS if input is file           |                                                               |

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: With DeepStream 7.1, the docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio track. Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features:

```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

### Reference
- DeepStream SDK Python: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
- Open CV: https://github.com/opencv/opencv
