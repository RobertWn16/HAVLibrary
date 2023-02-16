#include "NVENC.hpp"

#pragma comment (lib, "nvencodeapi")
static winrt::hresult NVENCStatHr(NVENCSTATUS nvenc_stat)
{
    if (nvenc_stat != NV_ENC_SUCCESS)
        return E_INVALIDARG;
    return S_OK;
}
NVENC::~NVENC()
{
}
winrt::hresult NVENC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    return winrt::hresult();
}

winrt::hresult NVENC::Encode(IFrame* inFrame)
{
    NVFrame* nv_frame = dynamic_cast<NVFrame*>(inFrame);
    if (nv_frame)
    {
        try
        {
            cuCtxPushCurrent(deviceCtx);

            CUDA_MEMCPY2D m = { 0 };
            m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            m.srcDevice = nv_frame->cuFrame;
            m.srcPitch = nvencFramePitch;
            m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            m.dstDevice = nvencFrame;
            m.dstPitch = nvencFramePitch;
            m.WidthInBytes = m.dstPitch;
            m.Height = 2160;
            CUresult res = cuMemcpy2D(&m);

            cuCtxPushCurrent(nullptr);

            NV_ENC_MAP_INPUT_RESOURCE nv_enc_map_res = { NV_ENC_MAP_INPUT_RESOURCE_VER };
            nv_enc_map_res.registeredResource = nvencRegisteredPtr;
            winrt::check_hresult(NVENCStatHr(nvenc.nvEncMapInputResource(nvencPtr, &nv_enc_map_res)));

            NV_ENC_PIC_PARAMS pic_params = { NV_ENC_PIC_PARAMS_VER };
            pic_params.version = NV_ENC_PIC_PARAMS_VER;
            pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
            pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
            pic_params.inputBuffer = nv_enc_map_res.mappedResource;
            pic_params.inputHeight = 2160;
            pic_params.inputWidth = 3840;
            pic_params.inputPitch = 0;
            pic_params.completionEvent = nvencCompletedFrame;
            pic_params.outputBitstream = nvencBistreamOutPtr;
            NVENCSTATUS stat =  nvenc.nvEncEncodePicture(nvencPtr, &pic_params);

            winrt::check_hresult(NVENCStatHr(nvenc.nvEncUnmapInputResource(nvencPtr, nv_enc_map_res.mappedResource)));
        } catch (winrt::hresult_error& err) {
            return err.code();
        }


    }
    return S_OK;
}

winrt::hresult NVENC::GetEncodedPacket(IPacket* outPkt)
{
    try
    {
        DWORD dwResult = WaitForSingleObject(nvencCompletedFrame, INFINITE);
        if (dwResult != WAIT_TIMEOUT) {
            NV_ENC_LOCK_BITSTREAM bitstream_lock = { NV_ENC_LOCK_BITSTREAM_VER };
            bitstream_lock.outputBitstream = nvencBistreamOutPtr;
            bitstream_lock.doNotWait = false;
            winrt::check_hresult(NVENCStatHr(nvenc.nvEncLockBitstream(nvencPtr, &bitstream_lock)));
            FFMPEGPacket* ffmpeg_pkt = dynamic_cast<FFMPEGPacket*>(outPkt);
            if (outPkt)
            {
                AVPacket* pkt = av_packet_alloc();
                pkt->stream_index = 0;
                pkt->data = reinterpret_cast<uint8_t*>(bitstream_lock.bitstreamBufferPtr);
                pkt->size = bitstream_lock.bitstreamSizeInBytes;
                pkt->dts = bitstream_lock.outputTimeStamp;
                pkt->pts = bitstream_lock.outputDuration;

                ffmpeg_pkt->RecievePacket(pkt);
                av_packet_free(&pkt);
            }
            winrt::check_hresult(NVENCStatHr(nvenc.nvEncUnlockBitstream(nvencPtr, bitstream_lock.outputBitstream)));
        }
    } catch (winrt::hresult_error& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult NVENC::ConfigureEncoder(ENCODER_DESC nvencDesc, CUcontext deviceContext)
{
    try
    {
        NVENCSTATUS stat = NV_ENC_SUCCESS;
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS nvencOpenPar = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
        uint32_t numOfGuids;
        GUID guids[5];
        GUID presetGuids[20];
        NV_ENC_TUNING_INFO nvencTunInfo;
        NV_ENC_PRESET_CONFIG nvencPresetConf = { NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER} };
        nvencPresetConf.presetCfg.version = NV_ENC_CONFIG_VER;

        winrt::check_pointer(&deviceContext);
        deviceCtx = deviceContext;

        winrt::check_hresult(NVENCStatHr(NvEncodeAPICreateInstance(&nvenc)));
        nvencOpenPar.apiVersion = NVENCAPI_VERSION;
        nvencOpenPar.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        nvencOpenPar.device = deviceContext;
        winrt::check_hresult(NVENCStatHr(nvenc.nvEncOpenEncodeSessionEx(&nvencOpenPar, &nvencPtr)));
        winrt::check_hresult(NVENCStatHr(nvenc.nvEncGetEncodePresetConfigEx(nvencPtr, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P3_GUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &nvencPresetConf)));

        NV_ENC_INITIALIZE_PARAMS nvencParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        memset(&nvencParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
        nvencParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
        nvencParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
        nvencParams.encodeWidth = 3840;
        nvencParams.encodeHeight = 2160;
        nvencParams.maxEncodeHeight = 2160;
        nvencParams.maxEncodeWidth = 3840;
        nvencParams.darWidth = 3840;
        nvencParams.darHeight = 2160;
        nvencParams.enableEncodeAsync = true;
        nvencParams.frameRateNum = 60;
        nvencParams.frameRateDen = 1;
        nvencParams.enablePTD = 1;
        nvencParams.reportSliceOffsets = 0;
        nvencParams.enableSubFrameWrite = 0;
        nvencPresetConf.presetCfg.frameIntervalP = 1;
        nvencParams.encodeConfig = &nvencPresetConf.presetCfg;
        nvencParams.encodeConfig->gopLength = NVENC_INFINITE_GOPLENGTH;

        winrt::check_hresult(NVENCStatHr(nvenc.nvEncInitializeEncoder(nvencPtr, &nvencParams)));

        cuCtxPushCurrent(deviceCtx);

        cuMemAllocPitch(&nvencFrame, &nvencFramePitch, 3840 * 4, 2160, 16);

        NV_ENC_REGISTER_RESOURCE nv_enc_resource = { NV_ENC_REGISTER_RESOURCE_VER };
        nv_enc_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        nv_enc_resource.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
        nv_enc_resource.resourceToRegister = (void*)nvencFrame;
        nv_enc_resource.width = 3840;
        nv_enc_resource.height = 2160;
        nv_enc_resource.pitch = nvencFramePitch;

        winrt::check_hresult(NVENCStatHr(nvenc.nvEncRegisterResource(nvencPtr, &nv_enc_resource)));
        nvencRegisteredPtr = nv_enc_resource.registeredResource;

        NV_ENC_CREATE_BITSTREAM_BUFFER nv_enc_bitstream;
        nv_enc_bitstream = {NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
        winrt::check_hresult(NVENCStatHr(nvenc.nvEncCreateBitstreamBuffer(nvencPtr, &nv_enc_bitstream)));
        nvencBistreamOutPtr = nv_enc_bitstream.bitstreamBuffer;

        nvencCompletedFrame = CreateEvent(NULL, FALSE, FALSE, NULL);
        winrt::check_pointer(nvencCompletedFrame);
        NV_ENC_EVENT_PARAMS eventParams = { NV_ENC_EVENT_PARAMS_VER };
        eventParams.completionEvent = nvencCompletedFrame;
        winrt::check_hresult(NVENCStatHr(nvenc.nvEncRegisterAsyncEvent(nvencPtr, &eventParams)));

        cuCtxPopCurrent(nullptr);

        } catch (winrt::hresult_error& err) {
            return err.code();
        }

    return S_OK;
}
