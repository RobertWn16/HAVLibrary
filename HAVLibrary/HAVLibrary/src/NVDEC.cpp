#include "NVDEC.hpp"
#include <iostream>
#include <fstream>
#include "NVENC.hpp"
static double GetFrameSizeCh(HVChroma chroma)
{
    switch (chroma)
    {
    case HV_CHROMA_FORMAT_420:
        return 1.5f;
    case HV_CHROMA_FORMAT_422:
        return 1.5f;
    case HV_CHROMA_FORMAT_444:
        return 3.0f;
    default:
        break;
    }
}

static winrt::hresult CUDAHr(cudaError_t cudaErr)
{
    if (cudaErr < cudaSuccess)
        return E_FAIL;

    return S_OK;
}


static winrt::hresult CUHr(CUresult cuRes)
{
    if (cuRes != CUDA_SUCCESS)
        return E_FAIL;

    return S_OK;
}

//Utils
static cudaVideoCodec HAVCONV(int havcodec)
{
    switch (havcodec)
    {
    case HV_CODEC_MPEG1:
        return cudaVideoCodec_MPEG1;
    case HV_CODEC_MPEG2:
        return cudaVideoCodec_MPEG2;
    case HV_CODEC_VC1:
        return cudaVideoCodec_VC1;
    case HV_CODEC_VP8:
        return cudaVideoCodec_VP8;
    case HV_CODEC_VP9:
        return cudaVideoCodec_VP9;
    case HV_CODEC_H264:
        return cudaVideoCodec_H264;
    case HV_CODEC_H265_420:
        return cudaVideoCodec_HEVC;
    case HV_CODEC_AV1:
        return cudaVideoCodec_AV1;
    case HV_CODEC_MJPEG:
        return cudaVideoCodec_JPEG;
    default:
        break;
    }
}

static cudaVideoChromaFormat HAVCHNV(int havchroma)
{
    switch (havchroma)
    {
    case HV_CHROMA_FORMAT_420:
        return cudaVideoChromaFormat_420;
    case HV_CHROMA_FORMAT_422:
        return cudaVideoChromaFormat_422;
    case HV_CHROMA_FORMAT_444:
        return cudaVideoChromaFormat_444;
    }
}

NVDEC::~NVDEC()
{
    cuCtxPushCurrent(deviceContext);

    if (cuParser) {
        cuvidDestroyVideoParser(cuParser);
        cuParser = nullptr;
    }

    if (cuDecoder) {
        cuvidDestroyDecoder(cuDecoder);
        cuParser = nullptr;
    }

    if (dec_bkbuffer) {
        cudaFree(reinterpret_cast<void*>(dec_bkbuffer));
        cuParser = nullptr;
    }

    cuCtxPopCurrent(nullptr);
}

winrt::hresult NVDEC::CreateParser(VIDEO_SOURCE_DESC desc)
{
    CUVIDPARSERPARAMS cuParserParams = { };

    cuParserParams.CodecType = HAVCONV(desc.codec);
    cuParserParams.pUserData = this;
    cuParserParams.ulMaxDisplayDelay = 2;
    cuParserParams.ulMaxNumDecodeSurfaces = GetNumDecodeSurfaces(cuParserParams.CodecType, desc.width, desc.heigth);
    cuParserParams.pfnSequenceCallback = reinterpret_cast<PFNVIDSEQUENCECALLBACK>(parser_sequence_callback);
    cuParserParams.pfnDecodePicture = reinterpret_cast<PFNVIDDECODECALLBACK>(parser_decode_picture_callback);
    cuParserParams.pfnDisplayPicture = reinterpret_cast<PFNVIDDISPLAYCALLBACK>(parser_display_picture_callback);
    cuParserParams.pfnGetOperatingPoint = reinterpret_cast<PFNVIDOPPOINTCALLBACK>(parser_get_operation_point_callback);

    winrt::check_hresult(CUHr(cuvidCreateVideoParser(&cuParser, &cuParserParams)));


    unsigned int cellSize = (desc.bitdepth - 8);
    cellSize = (cellSize) ? cellSize : 1;

    int32_t frameSize = GetFrameSizeCh(desc.chroma) * desc.width * desc.heigth * cellSize;
    winrt::check_hresult(CUDAHr(cudaMalloc((void**)&dec_bkbuffer, frameSize)));
    return S_OK;
}

winrt::hresult NVDEC::ConfigureDecoder(VIDEO_SOURCE_DESC vsrc_desc, IDev* dev)
{
    try {
        winrt::check_pointer(dev);
        DevNVIDIA* nv_dev = dynamic_cast<DevNVIDIA*>(dev);
        winrt::check_pointer(nv_dev);
        deviceContext = nv_dev->GetContext();
        winrt::check_hresult(IsSupported(vsrc_desc));
        winrt::check_hresult(CreateParser(vsrc_desc));
        cuVideo_desc = vsrc_desc;
    } catch (winrt::hresult_error& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult NVDEC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    if (desc.codec == HV_CODEC_MJPEG)
        return E_HV_CODEC_NOT_SUPPORTED;

    CUVIDDECODECAPS dec_caps;
    dec_caps.eCodecType = HAVCONV(desc.codec);
    dec_caps.eChromaFormat = HAVCHNV(desc.chroma);
    dec_caps.nBitDepthMinus8 = desc.bitdepth - 8;

    winrt::check_hresult(CUHr(cuCtxSetCurrent(deviceContext)));
    winrt::check_hresult(CUHr(cuvidGetDecoderCaps(&dec_caps)));
    winrt::check_hresult(CUHr(cuCtxSetCurrent(nullptr)));

    if (!dec_caps.bIsSupported)
        return E_HV_CODEC_NOT_SUPPORTED;

    return S_OK;
}

winrt::hresult NVDEC::Decode(IPacket* in, IFrame *out)
{
    try {
        PACKET_DESC pck_desc = { 0 };
        winrt::check_pointer(in);
        in->GetDesc(pck_desc);
        winrt::check_hresult(Decode(reinterpret_cast<unsigned char*>(pck_desc.data), pck_desc.size, pck_desc.timestamp, out));
    } catch (winrt::hresult_error& err) {
        return err.code();
    }
    return S_OK;
}

winrt::hresult NVDEC::Decode(unsigned char* buf, unsigned int length, unsigned int timestamp, IFrame* out)
{
    CUVIDSOURCEDATAPACKET cuPck = { 0 };
    cuPck.payload = reinterpret_cast<unsigned char*>(buf);
    cuPck.payload_size = length;
    cuPck.flags = CUVID_PKT_TIMESTAMP;
    cuPck.timestamp = timestamp;

    winrt::check_hresult(CUHr(cuvidParseVideoData(cuParser, &cuPck)));
    NVFrame* nv_frame = dynamic_cast<NVFrame*>(out);

    if (nv_frame) {
        unsigned int cellSize = (!(cuVideo_desc.bitdepth - 8)) ? 1 : (cuVideo_desc.bitdepth - 8);
        unsigned int frameSize = cuVideo_desc.width * cuVideo_desc.heigth * 3 * cellSize / 2.0f;
        winrt::check_hresult(CUDAHr(cudaMemcpy((void*)nv_frame->cuFrame, (void*)dec_bkbuffer, frameSize, cudaMemcpyDeviceToDevice)));
    }
    return S_OK;
}

int NVDEC::parser_decode_picture_callback(void* pUser, CUVIDPICPARAMS* pic)
{
    NVDEC* self = reinterpret_cast<NVDEC*>(pUser);
    winrt::check_pointer(self);
    CUresult res = cuvidDecodePicture(self->cuDecoder, pic);
    return 10;
}
int NVDEC::parser_sequence_callback(void* pUser, CUVIDEOFORMAT* fmt)
{
    NVDEC* self = reinterpret_cast<NVDEC*>(pUser);
    winrt::check_pointer(self);

    CUVIDDECODECREATEINFO create_info = { 0 };
    create_info.CodecType = fmt->codec;
    create_info.ChromaFormat = fmt->chroma_format;
    create_info.OutputFormat = (self->cuVideo_desc.bitdepth - 8) ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    create_info.bitDepthMinus8 = fmt->bit_depth_chroma_minus8;
    create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    create_info.ulNumOutputSurfaces = 3;
    create_info.ulNumDecodeSurfaces = GetNumDecodeSurfaces(fmt->codec, fmt->coded_width, fmt->coded_height);
    create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    create_info.vidLock = NULL;
    create_info.ulIntraDecodeOnly = 0;
    create_info.ulTargetWidth = fmt->coded_width;
    create_info.ulTargetHeight = fmt->coded_height;
    create_info.ulWidth = fmt->coded_width;
    create_info.ulHeight = fmt->coded_height;
    self->surfaceHeigth = fmt->coded_height;

    winrt::check_hresult(CUHr(cuCtxSetCurrent(self->deviceContext)));
    winrt::check_hresult(CUHr(cuvidCreateDecoder(&self->cuDecoder, &create_info)));
    winrt::check_hresult(CUHr(cuCtxSetCurrent(nullptr)));

    return create_info.ulNumDecodeSurfaces;
}
int NVDEC::parser_display_picture_callback(void* pUser, CUVIDPARSERDISPINFO* info) noexcept
{
    try
    {
        NVDEC* self = reinterpret_cast<NVDEC*>(pUser);
        winrt::check_pointer(self);

        CUdeviceptr dec_frame = 0;
        unsigned int pitch = 0;

        CUVIDPROCPARAMS vpp = { 0 };
        vpp.progressive_frame = info->progressive_frame;
        vpp.top_field_first = info->top_field_first;
        vpp.unpaired_field = (info->repeat_first_field < 0);
        vpp.second_field = info->repeat_first_field + 1;

        winrt::check_pointer(self->cuDecoder);
        winrt::check_pointer(self->deviceContext);

        CUVIDGETDECODESTATUS stat;
        bool checkState = false;
        if (self->cuVideo_desc.codec == HV_CODEC_H264 || self->cuVideo_desc.codec == HV_CODEC_H265_420) {
            checkState = true;
            winrt::check_hresult(CUHr(cuvidGetDecodeStatus(self->cuDecoder, info->picture_index, &stat)));
        }
        winrt::check_hresult(CUHr(cuCtxSetCurrent(self->deviceContext)));
        winrt::check_hresult(CUHr(cuvidMapVideoFrame(self->cuDecoder, info->picture_index, &dec_frame, &pitch, &vpp)));

        if (!checkState || stat.decodeStatus == cuvidDecodeStatus_Success || stat.decodeStatus == cuvidDecodeStatus_InProgress) {
            unsigned int cellSize = (self->cuVideo_desc.bitdepth - 8);
            cellSize = (cellSize) ? cellSize : 1;

            CUDA_MEMCPY2D m = { 0 };
            m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            m.srcDevice = dec_frame;
            m.srcPitch = pitch;
            m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            m.dstDevice = self->dec_bkbuffer;
            m.dstPitch = cellSize * self->cuVideo_desc.width;
            m.WidthInBytes = m.dstPitch;
            m.Height = self->cuVideo_desc.heigth;
            winrt::check_hresult(CUHr(cuMemcpy2D(&m)));
            m.srcDevice = dec_frame + m.srcPitch * self->surfaceHeigth;
            m.dstDevice = self->dec_bkbuffer + m.dstPitch * self->cuVideo_desc.heigth;
            m.Height = self->cuVideo_desc.heigth / 2;
            winrt::check_hresult(CUHr(cuMemcpy2D(&m)));

        }
        winrt::check_hresult(CUHr(cuvidUnmapVideoFrame(self->cuDecoder, dec_frame)));
        winrt::check_hresult(CUHr(cuCtxSetCurrent(nullptr)));

    } catch (winrt::hresult_error const& err) {
        std::wcout << HAV_LOG << "DisplayPictureCallback " << err.message().c_str() << std::endl;
    }

    return 0;
}

int NVDEC::parser_get_operation_point_callback(void* pUser, CUVIDOPERATINGPOINTINFO* opInfo)
{
    return 0;
}