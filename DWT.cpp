#include <iostream>
#include <cv.h>
#include<highgui.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
using namespace cv;


__global__ void kernel(int nWidth, int nHeight, int widthStep, char **dev_imageData)
{
    int nLayer = 1;
    int i, j,x, y, n;
    float fValue = 0;
    float fRadius = sqrt(2.0f);
    int nHalfW = nWidth / 2;
    int nHalfH = nHeight / 2;
    float **pData = new float*[nHeight];
    float *pRow = new float[nWidth];
    float *pColumn = new float[nHeight];

    for (j = 0; j < 3;j++)
    {
        for (i = 0; i < nHeight; i++)
        {
            pData[i] = (float*)(*(dev_imageData[j]) + widthStep*i);
        }
        //多层小波变换
        for (n = 0; n < nLayer; n++, nWidth /= 2, nHeight /= 2, nHalfW /= 2, nHalfH /= 2)//???
        {
            //水平变换，
            for (y = 0; y < nHeight; y++)
            {
                //奇偶分离
                //memcpy(pRow, pData[y], nWidth*sizeof(float));
                for (i = 0; i < nWidth; i++)//相当于memcpy
                {
                    pRow[i] = pData[y][i];
                }
                
                for (i = 0; i < nHalfW; i++) 
                {
                    x = i * 2;//x=0，2，4.......是偶数，x+1=1,3,5......是奇数
                    pData[y][i] = pRow[x];//y表示的是当前行，i=0,1,2,......74, pData[y][i]里存储的是偶数部分
                    pData[y][nHalfW + i] = pRow[x + 1];//nHalfW+i=75,76,......149,pData[y][nHalfW+i]里存储的是奇数部分
                }
                //提升小波变换
                for (i = 0; i < nHalfW - 1; i++)
                {
                    fValue = (pData[y][i] + pData[y][i + 1]) / 2; //用偶数序列预测奇数序列
                    pData[y][nHalfW + i] -= fValue;//用奇数序列减去偶数序列预测出的值，得到差值，并用该差值替换奇数序列，该差值是高通小波系数，反映的是图像的细节部分

                }
                //不明白为什么有下面这几句？？？？前两句是对预测最后的处理，后两句是对更新开始的处理
                fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
                pData[y][nWidth - 1] -= fValue;
                fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
                pData[y][0] += fValue;
                for (i = 1; i < nHalfW; i++)
                {
                    fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4;
                    pData[y][i] += fValue;
                }
                //频带系数，系数缩放，
                for (i = 0; i < nHalfW; i++)
                {
                    pData[y][i] *= fRadius;
                    pData[y][nHalfW + i] /= fRadius;
                }


            }

            //垂直变换
            for (x = 0; x < nWidth; x++)
            {
                //奇偶分离
                for (i = 0; i < nHalfH; i++)
                {
                    y = i * 2;
                    pColumn[i] = pData[y][x];
                    pColumn[nHalfH + i] = pData[y + 1][x];
                }
                for (i = 0; i < nHeight; i++)
                {
                    pData[i][x] = pColumn[i];
                }
                //提升小波变换
                for (i = 0; i < nHalfH - 1; i++)
                {
                    fValue = (pData[i][x] + pData[i + 1][x]) / 2;
                    pData[nHalfH + i][x] -= fValue;
                }
                fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
                pData[nHeight - 1][x] -= fValue;
                fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4;
                pData[0][x] += fValue;
                for (i = 1; i < nHalfH; i++)
                {
                    fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4;
                    pData[i][x] += fValue;
                }
                //频带系数
                for (i = 0; i < nHalfH; i++)
                {
                    pData[i][x] *= fRadius;
                    pData[nHalfH + i][x] /= fRadius;
                }
            }

        }
        delete[] pData;
        delete[] pRow;
        delete[] pColumn;

    }
}


void DWT(IplImage *pImage1, IplImage *pImage2, IplImage *pImage3, int nLayer)
{
    if (pImage1&&pImage2&&pImage3)
    {
        cudaError_t cudaStatus;
        int nWidth = pImage1->width;
        int nHeight = pImage1->height;
        int widthStep = pImage1->widthStep;
        //char *imageData[3] = { pImage1->imageData, pImage2->imageData, pImage3->imageData };
        //char *dev_imageData[3];
        char *h1 = pImage1->imageData;
        char *h2 = pImage2->imageData;
        char *h3 = pImage3->imageData;
        char *d1 = NULL;
        char *d2 = NULL;
        char *d3 = NULL;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            
        }


        cudaStatus = cudaMalloc((void**)&d1, nWidth*nHeight*sizeof(char));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");

        }
        cudaStatus = cudaMalloc((void**)&d2, nWidth*nHeight*sizeof(char));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");

        }
        cudaStatus = cudaMalloc((void**)&d3, nWidth*nHeight*sizeof(char));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");

        }

        cudaStatus = cudaMemcpy(d1, h1,  nWidth*nHeight*sizeof(char), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {

            fprintf(stderr, "cudaMemcpy failed!");

        }

        cudaStatus = cudaMemcpy(d2, h2, nWidth*nHeight*sizeof(char), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {

            fprintf(stderr, "cudaMemcpy failed!");

        }
        cudaStatus = cudaMemcpy(d3, h3, nWidth*nHeight*sizeof(char), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {

            fprintf(stderr, "cudaMemcpy failed!");

        }

        char *h_imageData[] = {d1,d2,d3};
        char **dev_imageData = NULL;

        cudaStatus = cudaMalloc((void***)&dev_imageData, sizeof(h_imageData));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");

        }

        cudaStatus = cudaMemcpy(dev_imageData, h_imageData, sizeof(h_imageData), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {

            fprintf(stderr, "cudaMemcpy failed!");

        }

        kernel << < 1, 300 >> >(nWidth, nHeight, widthStep, dev_imageData);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            
        }


        cudaStatus = cudaMemcpy(h_imageData, dev_imageData, sizeof(dev_imageData), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {

            fprintf(stderr, "cudaMemcpy failed!");

        }
 
        cudaFree(dev_imageData);
        cudaFree(d1);
        cudaFree(d2);
        cudaFree(d3);

        

    }

}


void main()
{

    int nLayer = 1;//小波变换层数
    IplImage *pSrc = cvLoadImage("lena.jpg", CV_LOAD_IMAGE_COLOR);//输入彩色图像
    CvSize size = cvGetSize(pSrc);//计算小波图像大小
    
    if ((pSrc->width >> nLayer) << nLayer != pSrc->width)  
    {
        size.width = ((pSrc->width >> nLayer) + 1) << nLayer;
    }
    if ((pSrc->height >> nLayer) << nLayer != pSrc->height)
    {
        size.height = ((pSrc->height >> nLayer) + 1) << nLayer;
    }
    IplImage *pWavlet = cvCreateImage(size, IPL_DEPTH_32F, pSrc->nChannels);//创建小波图像
    if (pWavlet)
    {
        //小波图像赋值
        cvSetImageROI(pWavlet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pSrc, pWavlet, 1, -128);
        cvResetImageROI(pWavlet);
        //彩色图像小波变换
        IplImage *pImage1 = cvCreateImage(cvGetSize(pWavlet), IPL_DEPTH_32F, 1);
        IplImage *pImage2 = cvCreateImage(cvGetSize(pWavlet), IPL_DEPTH_32F, 1);
        IplImage *pImage3 = cvCreateImage(cvGetSize(pWavlet), IPL_DEPTH_32F, 1);
        if (pImage1&&pImage2&&pImage3)
        {
            
            cvSetImageCOI(pWavlet, 1);
            cvCopy(pWavlet, pImage1, NULL);

            cvSetImageCOI(pWavlet, 2);
            cvCopy(pWavlet, pImage2, NULL);

            cvSetImageCOI(pWavlet, 3);
            cvCopy(pWavlet, pImage3, NULL);


            DWT(pImage1,pImage2,pImage3, nLayer);

            cvCopy(pImage1, pWavlet, NULL);
            cvCopy(pImage2, pWavlet, NULL);
            cvCopy(pImage3, pWavlet, NULL);


            cvSetImageCOI(pWavlet, 0);
            cvReleaseImage(&pImage1);
            cvReleaseImage(&pImage2);
            cvReleaseImage(&pImage3);
        }

        //小波变换图像
        cvSetImageROI(pWavlet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pWavlet, pSrc, 1, 128);
        cvResetImageROI(pWavlet);
        cvReleaseImage(&pWavlet);



    }

    //显示图像pSrc
    cvNamedWindow("yuhuan", 0);
    cvShowImage("yuhuan", pSrc);
    cvSaveImage("F:\lena2.jpg", pSrc);
    cvWaitKey(10000);
    cvReleaseImage(&pSrc);

}