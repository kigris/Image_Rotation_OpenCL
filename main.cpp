//
//  main.cpp
//  Lab3
//
//  Created by Adrian Daniel Bodirlau on 21/11/2022.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "CL/openCLutils.hpp"
#include "bmpfuncs.c"
#include <string>
#include <vector>
using std::string;
using namespace cv;
using std::cout;
using std::vector;

int main(int argc, const char* argv[]) {
    cl_platform_id* platforms;
    CLGetPlatforms(platforms);
    
    cl_device_id* devices;
    CLGetDevices(platforms[0], devices, CL_DEVICE_TYPE_GPU);
    cl_device_id device = devices[0];
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue commands = clCreateCommandQueue(context, device, 0, NULL);
    
    string programSourceStr = readFile((char*)"Lab3.2/source.CL");
    const char* programSource = programSourceStr.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, NULL, NULL);
    CLBuildProgram(&program, &device);
    
    // Reading one channel from the image and converting values to float
    Mat imgSrc = imread("Lab3.2/input.jpg", 0);
    imgSrc.convertTo(imgSrc, CV_32F);
    
    cl_image_format imgFormat;
    imgFormat.image_channel_order = CL_R;
    imgFormat.image_channel_data_type = CL_FLOAT;
    cl_image_desc imgDesc;
    imgDesc.image_width = imgSrc.cols;
    imgDesc.image_height = imgSrc.rows;
    imgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    
    cl_mem imgInBuff = clCreateImage(context, CL_MEM_READ_ONLY, &imgFormat, &imgDesc, NULL, NULL);
    cl_mem imgOutBuff = clCreateImage(context, CL_MEM_WRITE_ONLY, &imgFormat, &imgDesc, NULL, NULL);
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, NULL);
    
    size_t origin[3]{0,0,0};
    size_t region[3]{(size_t)imgSrc.cols,(size_t)imgSrc.rows,1};
    clEnqueueWriteImage(commands, imgInBuff, CL_TRUE, origin, region, 0, 0, imgSrc.ptr<float>(), 0, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "rotateImage", NULL);
    
    float sin = 0.087;
    float cos = 0.996;
    int xOffs = -300;
    int yOffs = -200;
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(imgInBuff), &imgInBuff);
    clSetKernelArg(kernel, 1, sizeof(imgOutBuff), &imgOutBuff);
    clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
    clSetKernelArg(kernel, 3, sizeof(sin), &sin);
    clSetKernelArg(kernel, 4, sizeof(cos), &cos);
    clSetKernelArg(kernel, 5, sizeof(xOffs), &xOffs);
    clSetKernelArg(kernel, 6, sizeof(yOffs), &yOffs);
    
    size_t globalWorkers[2]{(size_t)imgSrc.cols,(size_t)imgSrc.rows};
    clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    
    float* imgOutArray =(float*)malloc(imgSrc.cols*imgSrc.rows*sizeof(float));
    clEnqueueReadImage(commands, imgOutBuff, CL_TRUE, origin, region, 0, 0, imgOutArray, 0, NULL, NULL);
    
    Mat outputImage(imgSrc.rows,imgSrc.cols,imgSrc.type(), (unsigned*)imgOutArray);
    imwrite("Lab3.2/output.jpg", outputImage);
    
//    storeImage(imgOutArray, (char*)"Lab3.2/output2.bmp", imgSrc.rows, imgSrc.cols, (char*)"Lab3.2/input.bmp");

    return 0;
}
