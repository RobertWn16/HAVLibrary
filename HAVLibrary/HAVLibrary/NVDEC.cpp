#include "NVDEC.hpp"
#include <iostream>

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
    //cuParserParams.pfnGetOperatingPoint =

    CUresult res = cuvidCreateVideoParser(&cuParser, &cuParserParams);
    return S_OK;
}

winrt::hresult NVDEC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    CUVIDDECODECAPS dec_caps;
    dec_caps.eCodecType = HAVCONV(desc.codec);
    dec_caps.eChromaFormat = HAVCHNV(desc.chroma);
    dec_caps.nBitDepthMinus8 = desc.bitdepth - 8;
    CUresult cuResult = cuvidGetDecoderCaps(&dec_caps);

    if (!dec_caps.bIsSupported)
        return E_HV_CODEC_NOT_SUPPORTED;

    return S_OK;
}

winrt::hresult NVDEC::Decode(IFrame** out)
{
    CUVIDSOURCEDATAPACKET cuPck = { 0 };
    PACKET_DESC pck_desc = { 0 };

    vSource->Parse(&pck_desc);
    cuPck.payload = reinterpret_cast<unsigned char*>(pck_desc.data);
    cuPck.payload_size = pck_desc.size;
    cuPck.flags = 0;

    CUresult cuRes = cuvidParseVideoData(cuParser, &cuPck);

    printf("%d \n", cuRes);
    return S_OK;
}

int NVDEC::parser_decode_picture_callback(void* pUser, CUVIDPICPARAMS* pic)
{
    NVDEC* self = reinterpret_cast<NVDEC*>(pUser);

    CUresult res = cuvidDecodePicture(self->cuDecoder, pic);

    return 10;
}
int NVDEC::parser_sequence_callback(void* pUser, CUVIDEOFORMAT* fmt)
{
    NVDEC* self = reinterpret_cast<NVDEC*>(pUser);

    CUVIDDECODECREATEINFO create_info = { 0 };
    create_info.CodecType = fmt->codec;
    create_info.ChromaFormat = fmt->chroma_format;
    create_info.OutputFormat = (fmt->bit_depth_luma_minus8) ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    create_info.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
    create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    create_info.ulNumOutputSurfaces = 2;
    create_info.ulNumDecodeSurfaces = GetNumDecodeSurfaces(fmt->codec, fmt->coded_width, fmt->coded_height);
    create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    create_info.vidLock = NULL;
    create_info.ulIntraDecodeOnly = 0;
    create_info.ulTargetWidth = fmt->coded_width;
    create_info.ulTargetHeight = fmt->coded_height;
    create_info.ulWidth = fmt->coded_width;
    create_info.ulHeight = fmt->coded_height;

    cuCtxSetCurrent(self->deviceContext);
    CUresult res = cuvidCreateDecoder(&self->cuDecoder, &create_info);
    cuCtxSetCurrent(nullptr);

    return create_info.ulNumDecodeSurfaces;
}
int NVDEC::parser_display_picture_callback(void* pUser, CUVIDPARSERDISPINFO* info)
{
    NVDEC* self = reinterpret_cast<NVDEC*>(pUser);
    CUdeviceptr dec_frame = 0;
    CUdeviceptr dec_bkbuffer = 0;
    unsigned int pitch = 0;

    CUVIDPROCPARAMS vpp = { 0 };
    vpp.progressive_frame = info->progressive_frame;
    vpp.top_field_first = info->top_field_first + 1;
    vpp.unpaired_field = (info->repeat_first_field < 0);
    vpp.second_field = info->repeat_first_field + 1;

    cuCtxSetCurrent(self->deviceContext);
    CUresult res = cuvidMapVideoFrame(self->cuDecoder, info->picture_index, &dec_frame, &pitch, &vpp);

    cuvidUnmapVideoFrame(self->cuDecoder, dec_frame);
    cuCtxSetCurrent(nullptr);

    printf("Picture index %d", info->picture_index);

    return 0;
}