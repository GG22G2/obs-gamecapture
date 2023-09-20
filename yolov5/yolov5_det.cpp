#include "det_dll_export.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>


using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, bool &is_p6, float &gd, float &gw,
                std::string &img_dir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
        if (net.size() == 2 && net[1] == '6') {
            is_p6 = true;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

void
prepare_buffers(ICudaEngine *engine, float **gpu_input_buffer, float **gpu_output_buffer, float **cpu_output_buffer) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **) gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **gpu_buffers, float *output, int batchsize) {
    context.enqueue(batchsize, gpu_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, bool &is_p6, float &gd, float &gw, std::string &wts_name,
                      std::string &engine_name) {
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory *serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
    serialized_engine->destroy();
}


void
deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

// Deserialize the engine from file
IRuntime *runtime = nullptr;
ICudaEngine *engine = nullptr;
IExecutionContext *context = nullptr;
cudaStream_t stream;
// Prepare cpu and gpu buffers
float *gpu_buffers[2];
float *cpu_output_buffer = nullptr;

int detect_init(const char *engine_name) {
    //engine_name = "G:\\dataset\\csgo\\yolov5m_csgo.engine";
    std::string engine1 = std::string(engine_name);

    deserialize_engine(engine1, &runtime, &engine, &context);

    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);
    return 1;
}

std::vector<Detection>
inference_inner(unsigned char *data, int cols, int rows, bool gpuDate = false, bool bgraIma = false) {
    int dst_size = kInputW * kInputH * 3;
    //   auto start = std::chrono::system_clock::now();

    if (gpuDate) {
        if (bgraIma) {
   //         auto start = std::chrono::system_clock::now();
            cuda_preprocess3(data, cols, rows, &gpu_buffers[0][dst_size * 0], kInputW, kInputH, stream);
 //           auto end = std::chrono::system_clock::now();
//            std::cout << "cuda_preprocess3 time: "
//                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms"
//                      << std::endl;
        } else {
    //        auto start = std::chrono::system_clock::now();
            cuda_preprocess2(data, cols, rows, &gpu_buffers[0][dst_size * 0], kInputW, kInputH, stream);
  //          auto end = std::chrono::system_clock::now();
//            std::cout << "cuda_preprocess2 time: "
//                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms"
//                      << std::endl;
        }
    } else {
        cuda_preprocess(data, cols, rows, &gpu_buffers[0][dst_size * 0], kInputW, kInputH, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 预测
    infer(*context, stream, (void **) gpu_buffers, cpu_output_buffer, kBatchSize);
    // NMS
    //std::vector<Detection> res_batch;
    std::vector<Detection> res_batch;
    //  batch_nms(res_batch, cpu_output_buffer, 1, kOutputSize, kConfThresh, kNmsThresh);
    nms(res_batch, &cpu_output_buffer[0 * kOutputSize], kConfThresh, kNmsThresh);

    return res_batch;
}

float result[120];

//= (float *) malloc(6 * 20 * 4)
//预测
float *detect_inference(unsigned char *data, int cols, int rows) {
    //todo 这个res_batch对应的内存会自动释放吗？ 栈上创建对象会被拷贝几次？
    std::vector<Detection> res_batch = inference_inner(data, cols, rows);
    int t = 0;
    for (size_t j = 0; j < res_batch.size(); j++, t += 6) {
        auto res = res_batch[j];
        cv::Rect r = get_rect(cols, rows, res.bbox);
        result[t] = r.x;
        result[t + 1] = r.y;
        result[t + 2] = r.width;
        result[t + 3] = r.height;
        result[t + 4] = res.conf;
        result[t + 5] = res.class_id;
    }
    for (size_t i = t; i < 120; i++) {
        result[i] = 0;
    }
    return result;
}

#define GPU_BGRA_IMG 1
#define GPU_BGR_IMG 0

//data直接就是一个gpu上的指针，指向一块像素数据，
// type = 0  bgr   type=1  bgra
float *detect_inferenceGpuData(unsigned char *data, int cols, int rows, int type) {
    //todo 这个res_batch对应的内存会自动释放吗？ 栈上创建对象会被拷贝几次？
    std::vector<Detection> res_batch = inference_inner(data, cols, rows, true, type == GPU_BGRA_IMG);
    int t = 0;
    for (size_t j = 0; j < res_batch.size(); j++, t += 6) {
        auto res = res_batch[j];
        cv::Rect r = get_rect(cols, rows, res.bbox);
        result[t] = r.x;
        result[t + 1] = r.y;
        result[t + 2] = r.width;
        result[t + 3] = r.height;
        result[t + 4] = res.conf;
        result[t + 5] = res.class_id;
    }
    for (size_t i = t; i < 120; i++) {
        result[i] = 0;
    }
    return result;
}


void detect_release() {
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}


int main(int argc, char **argv) {
    cudaSetDevice(kGpuId);

    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;

    if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr
                << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file"
                << std::endl;
        std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
        return 0;
    }

    detect_init(engine_name.c_str());


    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }


    cv::Mat img = cv::imread(img_dir + "/" + file_names[0]);


    while (false) {
        cv::Mat temp = img.clone();
        auto start = std::chrono::system_clock::now();
        float *result = detect_inference(img.ptr(), img.cols,
                                         img.rows);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
        temp.release();
    }

    unsigned char *data;
    cudaMalloc(&data, 864 * 416 * 4);


    cv::Mat bgrImg = cv::imread(
            R"(G:\kaifa_environment\code\clion\csgo-util\cmake-build-debug\yolov5\bin\img\1-51-5.png)");

    cv::resize(bgrImg, bgrImg, cv::Size(864, 416));
    cv::cvtColor(bgrImg, bgrImg, cv::COLOR_BGR2BGRA);

    //cv::imshow("211",bgrImg);
    //cv::waitKey(5000);

    cudaError result = cudaMemcpy(data, bgrImg.ptr(), 864 * 416 * 4, cudaMemcpyHostToDevice);
    cv::Mat temp = bgrImg.clone();
    for (int i = 0; i < 1000; ++i) {

        auto start = std::chrono::system_clock::now();
        float *result = detect_inferenceGpuData(data, 864,
                                                416,GPU_BGRA_IMG);
        std::cout << result[0] << std::endl;
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        cv::Rect r;
        r.x = (int) ((0 + result[0]));
        r.y = (int) ((0 + result[0 + 1]));
        r.width = (int) (result[0 + 2]);
        r.height = (int) (result[0 + 3]);

        // r = get_rect(img, res[j].bbox);

//        cv::rectangle(temp, r, cv::Scalar(0x27, 0xC1, 0x36), 1);
//        //cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
//
//        cv::imwrite("test.png"  ,temp);

        //  temp.release();
    }



    unsigned char *bgr_data_device;
    cudaMalloc(&bgr_data_device, 864 * 416 * 3);


    bgrImg = cv::imread(
            R"(G:\kaifa_environment\code\clion\csgo-util\cmake-build-debug\yolov5\bin\img\1-51-5.png)");

    cv::resize(bgrImg, bgrImg, cv::Size(864, 416));
    cudaMemcpy(bgr_data_device, bgrImg.ptr(), 864 * 416 * 3, cudaMemcpyHostToDevice);
    bgrImg.release();
    for (int i = 0; i < 1000; ++i) {
        float *result = detect_inferenceGpuData(bgr_data_device, 864,416,GPU_BGR_IMG);
        std::cout << result[0] << std::endl;
    }









    detect_release();


    return 0;
}

