#include "cfldwp_wrap.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "spdlog/spdlog.h"
#include <assert.h>
#include <iostream>
#include "cuda.h"
#include "cudaResize.h"
#include "cudaMappedMemory.h"
#include <thread>

using namespace std;


#ifdef _MSC_VER
#pragma comment(lib,"cfldwp2.lib")
#endif
namespace cfldwp{

#ifdef __GNUC__
#define CFLDWP_API  extern
#else
#define CFLDWP_API __declspec(dllimport)
#endif

    namespace detail{
        extern "C"{
            struct ErrInfo
            {
                int code;
                char errmsg[256 - 4];
            };

            CFLDWP_API int load_mod_cfg_json(const char* json, ErrInfo* pinfo);

            CFLDWP_API void wait_init_done();

            CFLDWP_API void* mod_by_name(const char* name);

            CFLDWP_API int mod_num_inputs(void* hdl);

            CFLDWP_API int mod_num_outputs(void* hdl);



            CFLDWP_API LayerDims mod_init_inp_dims(void* hdl, int idx, ErrInfo* pinfo);

            struct LayerDataRaw
            {
                int n, c, h, w;
                float* data;
                intptr_t idx_or_sz;
            };

            CFLDWP_API int mod_sync_proc(void* hdl, LayerDataRaw* pLdr, int nLdr, const char* outl_tok, int nol, LayerDataRaw* pOutlayerDatas, ErrInfo* pinfo);

            CFLDWP_API void free_outlayerdatas(LayerDataRaw* pOld, int n);


            CFLDWP_API int mat_convert_scale(void* data_u, int w, int h, int c, void* data_f, float alpha, float beta, float c1, float c2, float c3, float c4);

            CFLDWP_API int mat_8u_rgb2bgr(void* data_u, int w, int h, int c, void* data_out, int on);

            CFLDWP_API LayerDataRaw mat_imread(const char* fpath, int flag);

            CFLDWP_API void mat_free(LayerDataRaw* pm);
        }
    }

    shared_ptr<CflTaskPool> mod(const string& name)
    {
        void* hdl = detail::mod_by_name(name.data());
        if (!hdl) {
            throw Exception(fmt::format("cannot find mod of name '{}'", name));
        }
        return std::shared_ptr<CflTaskPool>(new CflTaskPool(hdl));
    }

    //////////////////////////////////////////////////////////////////////////
    CflTaskPool::CflTaskPool(void* v) : p(v), m_pixMean({}), m_pixScale(1.f)
    {
        //   int gpu_count = cv::gpu::getCudaEnabledDeviceCount();
        //   std::cout << "gpu count is: " << gpu_count << std::endl;
        //  cv::gpu::setDevice(0);
    }

    CflTaskPool::~CflTaskPool()
    {
    }

    void CflTaskPool::set_pixel_mean(cv::Scalar mean)
    {
        m_pixMean = mean;
    }

    void CflTaskPool::set_pixel_scale(float sc) {
        m_pixScale = sc;
    }

    int CflTaskPool::num_inputs() const {
        return detail::mod_num_inputs(p);
    }

    int CflTaskPool::num_outputs() const {
        return detail::mod_num_outputs(p);
    }

    LayerDims CflTaskPool::input_dim(size_t idx) const {
        detail::ErrInfo ei = {};
        auto ret = detail::mod_init_inp_dims(p, (int)idx, &ei);
        if (ei.code) {
            throw Exception(fmt::format("get input_dim failed: {}", ei.errmsg));
        }
        return ret;
    }

    void CflTaskPool::gpuResize(cv::Mat const& input, cv::Size const& dsize, cv::Mat& output) {
        clock_t t_start, t_end;
        t_start = clock();

        clock_t t_start1, t_end1;
        t_start1 = clock();

        cv::gpu::GpuMat gpu_input_image;

        t_end1 = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "1111111111111111111 GpuMat: gpu_input_image init - execution time =" << (double)(t_end1 - t_start1) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;

        clock_t t_start2, t_end2;
        t_start2 = clock();

        cv::gpu::GpuMat gpu_output_image;

        t_end2 = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "222222222222222222 GpuMat: gpu_output_image init - execution time =" << (double)(t_end2 - t_start2) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


        clock_t t_start3, t_end3;
        t_start3 = clock();

        cv::Mat copy_input(input);

        t_end3 = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "333333333333333333 copy_input init - execution time =" << (double)(t_end3 - t_start3) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


        clock_t t_start4, t_end4;
        t_start4 = clock();

        cv::gpu::registerPageLocked(copy_input);

        gpu_input_image.upload(input);

        t_end4 = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "4444444444444444444 gpuInImage.upload - execution time =" << (double)(t_end4 - t_start4) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


        clock_t t_start5, t_end5;
        t_start5 = clock();

        cv::gpu::resize(gpu_input_image, gpu_output_image, dsize);

        t_end5 = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "55555555555555555555 gpu::resize - execution time =" << (double)(t_end5 - t_start5) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


        clock_t t_start6, t_end6;
        t_start6 = clock();

        gpu_output_image.download(output);
        cv::gpu::unregisterPageLocked(copy_input);

        t_end6 = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "666666666666666666666 gpu::resize: gpuInImage.download - execution time =" << (double)(t_end6 - t_start6) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


        t_end = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "------------------------------ resize - execution time =" << (double)(t_end - t_start) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;

    }

    cv::Mat CflTaskPool::image_process_1(const cv::Mat &img, int nc, int w, int h, float scale, cv::Scalar subval) {
        cv::Mat sample_resize, sample, sample_float;

      //  cv::imwrite("/home/software/tensorrt/algorithm/algo_detect_track/sdk_src/algo_sdk/img.jpg", img);

        if (img.channels() != 4 && img.channels() != 3 && img.channels() != 1) {
            throw Exception(fmt::format("invalid img.channels() : {}", img.channels()));
        }
        if (img.depth() == CV_8U) {
            if (w != img.cols || h != img.rows) {
                clock_t t_start, t_end;
                t_start = clock();

                size_t isizeOfImage = img.step[0] * img.rows;

                if (h_resize_input == nullptr){
                    if (!cudaAllocMapped((void**)&h_resize_input, (void**)&d_resize_input, isizeOfImage)) {
                        exit(-1);
                    }
                }

                clock_t t_start8, t_end8;
                t_start8 = clock();

                memcpy(h_resize_input, img.data, isizeOfImage);
             //   uchar* temp_h_resize = img.data;
               // CUDA(cudaHostRegister(temp_h_resize, isizeOfImage, cudaHostRegisterMapped));
              //  cudaHostGetDevicePointer(&d_resize_input, temp_h_resize, 0);

                t_end8 = clock();
                std::cout << "[" << this_thread::get_id() << "]" << "pre_imgs: resize: 1  h_resize memcpy, isizeOfImage is " << isizeOfImage << " - execution time =" << (double)(t_end8 - t_start8) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;

                size_t osizeOfImage = w*h * 3;
                if (h_resize_output == nullptr) {
                    if (!cudaAllocMapped((void**)&h_resize_output, (void**)&d_resize_output, osizeOfImage)) {
                        exit(-1);
                    }
                }

                clock_t t_start3, t_end3;
                t_start3 = clock();

                auto error = cudaResizeRGB(d_resize_input, img.cols, img.rows, d_resize_output, w, h);
                cudaStreamSynchronize(NULL);

                CUDA(error);

                t_end3 = clock();
                std::cout << "[" << this_thread::get_id() << "]" << "pre_imgs: resize: 2  cudaResizeRGB  - execution time =" << (double)(t_end3 - t_start3) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;

                clock_t t_start4, t_end4;
                t_start4 = clock();

                sample_resize = cv::Mat(h, w, CV_8UC3, h_resize_output);
            //    cudaHostUnregister(temp_h_resize);


                t_end4 = clock();
                std::cout << "[" << this_thread::get_id() << "]" << "pre_imgs: resize: 3  cv::Mat  - execution time =" << (double)(t_end4 - t_start4) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


                t_end = clock();
                std::cout << "[" << this_thread::get_id() << "]" << "pre_imgs: resize: total - execution time =" << (double)(t_end - t_start) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;

               //  cv::imwrite("/home/software/tensorrt/algorithm/algo_detect_track/sdk_src/algo_sdk/resize.jpg", sample_resize);

            }

            if (nc == 3) {
                clock_t t_start, t_end;
                t_start = clock();
                if (img.channels() == 4)
                    cv::cvtColor(sample_resize, sample, CV_BGRA2RGB);
                else if (img.channels() == 1)
                    cv::cvtColor(sample_resize, sample, CV_GRAY2RGB);
                else
                    cv::cvtColor(sample_resize, sample, CV_BGR2RGB);

                t_end = clock();
                std::cout << "[" << this_thread::get_id() << "]" << "pre_imgs: `````````````` cvtColor - execution time =" << (double)(t_end - t_start) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


            }
            else {
                if (img.channels() == 4)
                    cv::cvtColor(sample_resize, sample, CV_BGRA2GRAY);
                else if (img.channels() == 3)
                    cv::cvtColor(sample_resize, sample, CV_BGR2GRAY);
                else //==1
                    sample = sample_resize;
            }

            clock_t t_start12, t_end12;
            t_start12 = clock();

            sample.convertTo(sample_float, CV_32FC(nc), scale);

            t_end12 = clock();
            std::cout << "[" << this_thread::get_id() << "]" << "pre_imgs: %12121212 convertTo - execution time =" << (double)(t_end12 - t_start12) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;
        }
        else { //depth = 32
            if (w != sample.cols || h != sample.rows) {
                cv::resize(sample, sample_float, cv::Size(w, h));
            }
            else {
                sample_float = sample;
            }
            if (scale != 1.0)
                sample_float *= scale;
        }

        return sample_float;
    }

    vector<float> CflTaskPool::prep_ims(const vector<cv::Mat>& input_ims, int c, int w, int h)
    {
        if (c != 1 && c != 3) {
            throw Exception("'nchannels' must be 1 or 3");
        }
        vector<float> outdata(input_ims.size()*c*w*h);
        float* input_data = outdata.data();
        for (auto& mat : input_ims) {

            clock_t t_start, t_end;
            t_start = clock();

            cv::Mat m_float = image_process_1(mat, c, w, h, m_pixScale, m_pixMean);

            t_end = clock();
            std::cout << "[" << this_thread::get_id() << "]" << "pre_imgs: ~~~~~image_process_1  - execution time =" << (double)(t_end - t_start) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;


            if (m_float.channels() != c) {
                throw Exception(fmt::format("channels of input image is mismatched against request: {}->{}", m_float.channels(), c));
            }

            if (c == 3) {
                clock_t t_start, t_end;
                t_start = clock();

                vector<cv::Mat> inp_mats;
                for (int i = 0; i < 3; ++i) {
                    cv::Mat channel(h, w, CV_32FC1, input_data);
                    inp_mats.push_back(channel);
                    input_data += w * h;
                }
                cv::split(m_float, inp_mats);

                t_end = clock();
                std::cout << "[" << this_thread::get_id() << "]" <<  "           channel split  - execution time =" << (double)(t_end - t_start) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;

            }
            else {
                memcpy(input_data, m_float.data, sizeof(float)*w*h);
                input_data += w * h;
            }
        }
        return outdata;
    }

    vector<OutLayerData> CflTaskPool::sync_proc_i0(const vector<cv::Mat>& input_ims, const vector<string>& out_layer_toks, cv::Size newsz /* = */)
    {
        auto idim = input_dim(0);
        int c = idim.c, w = idim.w, h = idim.h;
        if (newsz.width > 0 && newsz.height > 0) {
            w = newsz.width;
            h = newsz.height;
        }

        clock_t t_start, t_end;
        t_start = clock();

        vector<float> outdata = prep_ims(input_ims, c, w, h);

        t_end = clock();
        std::cout << "[" << this_thread::get_id() << "]" << ">>>>>>>>>>>>>prep_ims - execution time =" << (double)(t_end - t_start) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;
       
        
        //return vector<OutLayerData>();
        
        auto rets = sync_proc_i0_raw(LayerDims{ int(input_ims.size()), c, h, w }, outdata.data(), out_layer_toks);

        return rets;
    }

#include <sys/time.h>
    static auto freq = cv::getTickFrequency();
    long tickCount(){
        static struct timeval tv;
        gettimeofday(&tv, NULL);
        return  tv.tv_sec * 1000 * 1000 + tv.tv_usec;
    }

    vector<OutLayerData> CflTaskPool::sync_proc_i0_raw(LayerDims i0_dim, const float* input_data, const vector<string>& out_layer_toks)
    {
        detail::LayerDataRaw ldr = {};
        ldr.idx_or_sz = 0;
        ldr.c = i0_dim.c;
        ldr.w = i0_dim.w;
        ldr.h = i0_dim.h;
        ldr.n = i0_dim.n;
        ldr.data = const_cast<float*>(input_data);
        string outtokbuf;
        for (auto& s : out_layer_toks) {
            outtokbuf.append(s);
            outtokbuf.append(1, '\0');
        }
        detail::ErrInfo ei = {};
        vector<detail::LayerDataRaw> outlayers(out_layer_toks.size());

        clock_t t_start, t_end;
        t_start = clock();

        if (detail::mod_sync_proc(p, &ldr, 1, outtokbuf.data(), int(out_layer_toks.size()), outlayers.data(), &ei)) {
            throw Exception(fmt::format("sync_proc_i0 failed: {}", ei.errmsg));
        }

        t_end = clock();
        std::cout << "[" << this_thread::get_id() << "]" << "<<<============mod_sync_proc - execution time =" << (double)(t_end - t_start) * 1000 / CLOCKS_PER_SEC << "  ms" << std::endl;

        vector<OutLayerData> old(outlayers.size());
        for (size_t i = 0; i < old.size(); i++) {
            auto& d = old[i]; auto &s = outlayers[i];
            d.dim = { s.n, s.c, s.h, s.w };
            d.data.assign(s.data, s.data + s.idx_or_sz);
        }
        detail::free_outlayerdatas(outlayers.data(), outlayers.size());
        return old;
    }
    ///-=-------------------------------------------------////
    void load_mod_cfg_json(const string& json)
    {
        //_putenv_s("GLOG_minloglevel","2");
        detail::ErrInfo ei = {};
        int rt = detail::load_mod_cfg_json(json.data(), &ei);
        if (rt) {
            throw Exception(fmt::format("load_mod_cfg_json error: {},{}", ei.code, ei.errmsg));
        }
    }

    void wait_init_done()
    {
        detail::wait_init_done();
    }
}////END of namespace /////
