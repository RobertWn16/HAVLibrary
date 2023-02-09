#include "NVJpegDecoder.hpp"

constexpr int NO_OF_BGR_CHANNELS = 3;

winrt::hresult CUDAHr(cudaError_t cudaErr)
{
    if (cudaErr < cudaSuccess)
        return E_FAIL;

    return S_OK;
}

NVJpegDecoder::~NVJpegDecoder()
{
    if (nvjpegImage.channel[0]) {
        cudaFree(nvjpegImage.channel[0]);
        nvjpegImage.channel[0] = nullptr;
    }

    if (nvjpegState) {
        nvjpegJpegStateDestroy(nvjpegState);
        nvjpegState = nullptr;
    }

    if (nvHandle) {
        nvjpegDestroy(nvHandle);
        nvHandle = nullptr;
    }

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

winrt::hresult NVJpegDecoder::IsSupported(VIDEO_SOURCE_DESC video_src_desc)
{
    if (video_src_desc.codec != HV_CODEC_MJPEG)
        return E_HV_CODEC_NOT_SUPPORTED;

    return S_OK;
}

winrt::hresult NVJpegDecoder::Decode(IFrame* out)
{
    PACKET_DESC pck_desc = { 0 };
    NVFrame* nv_frame = dynamic_cast<NVFrame*>(out);
    try {
        winrt::check_pointer(vSource);
        winrt::check_hresult(vSource->Parse(&pck_desc));
        winrt::check_hresult(NVJPEGHr(nvjpegDecode(nvHandle, nvjpegState, reinterpret_cast<const unsigned char*>(pck_desc.data), pck_desc.size, NVJPEG_OUTPUT_BGRI, &nvjpegImage, stream)));
        winrt::check_hresult(CUDAHr(cudaMemcpy(reinterpret_cast<void*>(nv_frame->cuFrame), nvjpegImage.channel[0], NO_OF_BGR_CHANNELS * width * height, cudaMemcpyDeviceToDevice)));
    }catch (winrt::hresult_error const& err) {
        std::wcout << HAV_LOG << L"HAVComponent NVJPEG " << err.message().c_str() << std::endl;
        return err.code();
    }
    return S_OK;
}

winrt::hresult NVJpegDecoder::CreateNVJpeg(VIDEO_SOURCE_DESC video_source_desc) noexcept
{
    try {
        width = video_source_desc.width;
        height = video_source_desc.heigth;
        winrt::check_hresult(NVJPEGHr(nvjpegCreateSimple(&nvHandle)));
        winrt::check_hresult(NVJPEGHr(nvjpegJpegStateCreate(nvHandle, &nvjpegState)));
        winrt::check_hresult(CUDAHr(cudaMalloc(reinterpret_cast<void**>(&nvjpegImage.channel[0]), NO_OF_BGR_CHANNELS * width * height)));
        nvjpegImage.pitch[0] = 3 * width;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }catch (winrt::hresult_error const& err) {
        std::wcout << HAV_LOG << L"HAVComponent NVJPEG " << err.message().c_str() << std::endl;
        return err.code();
    }

    return S_OK;
}
