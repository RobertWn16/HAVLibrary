#include "NVENC.hpp"

#pragma comment (lib, "nvencodeapi")

static GUID HVPresetNVENCPreset(HVEncoderPreset hvPreset)
{
    switch (hvPreset)
    {
    case HV_ENCODER_ULTRA_LOW_QUALITY_H264:
        return NV_ENC_PRESET_P3_GUID;
        break;
    case HV_ENCODER_LOW_QUALITY_H264:
        return NV_ENC_PRESET_P4_GUID;
        break;
    case HV_ENCODER_MEDIUM_QUALITY_H264:
        return NV_ENC_PRESET_P5_GUID;
        break;
    case HV_ENCODER_HIGH_QUALITY_H264:
        return NV_ENC_PRESET_P6_GUID;
        break;
    default:
        break;
    }
}

static GUID HVCodecNVENCodec(HVCodec hvCodec)
{
    switch (hvCodec)
    {
    case HV_CODEC_H264:
        return NV_ENC_CODEC_H264_GUID;
    case HV_CODEC_H265_420:
    case HV_CODEC_H265_444:
        return NV_ENC_CODEC_HEVC_GUID;
    default:
        break;
    }
}
static winrt::hresult NVENCStatHr(NVENCSTATUS nvenc_stat)
{
    if (nvenc_stat != NV_ENC_SUCCESS)
        return E_INVALIDARG;
    return S_OK;
}
NVENC::~NVENC()
{
    SetEvent(nvencInternalEvents[NVENC_SESSION_CLOSED]);
}
winrt::hresult NVENC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    return S_OK;
}

winrt::hresult NVENC::Encode(IFrame* inFrame)
{
    NVFrame* nv_frame = dynamic_cast<NVFrame*>(inFrame);
    if (nv_frame)
    {
        try
        {
            cuCtxPushCurrent(nvencCtx);

            FRAME_OUTPUT_DESC nvframeDesc = { 0 };
            nv_frame->GetDesc(nvframeDesc);

            nvencTxInIndex = (nvencTxInIndex) % nvencDesc.num_in_back_buffers;
            nvencTxOutIndex = (nvencTxOutIndex) % nvencDesc.num_out_back_buffers;
            nvencCompletionRefQueue.push(nvencCompletedFrame[nvencTxOutIndex]);
            nvencBitstreamRefQueue.push(nvencBistreamOutPtr[nvencTxOutIndex]);

            CUDA_MEMCPY2D m = { 0 };
            m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            m.srcDevice = nv_frame->cuFrame;
            m.srcPitch = nvencFramePitch;
            m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            m.dstDevice = nvencFrame[nvencTxInIndex];
            m.dstPitch = nvencFramePitch;
            m.WidthInBytes = m.dstPitch;
            m.Height = nvframeDesc.height;
            CUresult res = cuMemcpy2D(&m);

            cuCtxPushCurrent(nullptr);

            NV_ENC_PIC_PARAMS pic_params = { NV_ENC_PIC_PARAMS_VER };
            pic_params.version = NV_ENC_PIC_PARAMS_VER;
            pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
            pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
            pic_params.inputBuffer = nvencMappedResource[nvencTxInIndex];
            pic_params.inputHeight = nvframeDesc.height;
            pic_params.inputWidth = nvframeDesc.width;
            pic_params.inputPitch = 0;
            pic_params.encodePicFlags = NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
            pic_params.completionEvent = nvencCompletedFrame[nvencTxOutIndex];
            pic_params.outputBitstream = nvencBistreamOutPtr[nvencTxOutIndex];
            NVENCSTATUS stat =  nvenc.nvEncEncodePicture(nvencPtr, &pic_params);

            nvencTxInIndex++;
            nvencTxOutIndex++;

            SetEvent(nvencInternalEvents[NVENC_SAFE_GRAB]);
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
        DWORD dwResults = WaitForMultipleObjects(2, nvencInternalEvents, false, 1000);

        if (dwResults == WAIT_OBJECT_0) {
            cuCtxPushCurrent(nvencCtx);
            DWORD dwResult = WaitForSingleObject(nvencCompletionRefQueue.front(), INFINITE);
            if (dwResult != WAIT_TIMEOUT) {
                NV_ENC_LOCK_BITSTREAM bitstream_lock = { NV_ENC_LOCK_BITSTREAM_VER };
                bitstream_lock.outputBitstream = nvencBitstreamRefQueue.front();
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

                nvencBitstreamRefQueue.pop();
                nvencCompletionRefQueue.pop();
            }
            cuCtxPushCurrent(nullptr);
        }
        if (dwResults == WAIT_TIMEOUT)
            return E_FAIL;

        if (dwResults == WAIT_OBJECT_0 + 1)
            return E_FAIL;

    } catch (winrt::hresult_error& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult NVENC::ConfigureEncoder(ENCODER_DESC nvDesc, CUcontext deviceContext)
{
    try
    {
        this->nvencDesc = nvDesc;
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS nvencOpenPar = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
        NV_ENC_PRESET_CONFIG nvencPresetConf = { NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER} };

        nvencPresetConf.presetCfg.version = NV_ENC_CONFIG_VER;

        winrt::check_pointer(&deviceContext);
        nvencCtx = deviceContext;

        winrt::check_hresult(NVENCStatHr(NvEncodeAPICreateInstance(&nvenc)));
        nvencOpenPar.apiVersion = NVENCAPI_VERSION;
        nvencOpenPar.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        nvencOpenPar.device = deviceContext;
        winrt::check_hresult(NVENCStatHr(nvenc.nvEncOpenEncodeSessionEx(&nvencOpenPar, &nvencPtr)));
        winrt::check_hresult(NVENCStatHr(nvenc.nvEncGetEncodePresetConfigEx(nvencPtr, HVCodecNVENCodec(nvencDesc.codec), HVPresetNVENCPreset(nvencDesc.preset), NV_ENC_TUNING_INFO_HIGH_QUALITY, &nvencPresetConf)));

        NV_ENC_INITIALIZE_PARAMS nvencParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        memset(&nvencParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
        nvencParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
        nvencParams.encodeGUID = HVCodecNVENCodec(nvencDesc.codec);
        nvencParams.encodeWidth = nvencDesc.encoded_width;
        nvencParams.encodeHeight = nvencDesc.encoded_height;
  
        nvencParams.maxEncodeHeight = nvencDesc.max_encoded_height;
        nvencParams.maxEncodeWidth = nvencDesc.max_encoded_width;
        nvencParams.darWidth = nvencDesc.encoded_width;
        nvencParams.darHeight = nvencDesc.encoded_height;
        nvencParams.enableEncodeAsync = true;
        nvencParams.frameRateNum = nvencDesc.framerate_num;
        nvencParams.frameRateDen = nvencDesc.framerate_den;
        nvencParams.enablePTD = 1;
        nvencParams.reportSliceOffsets = 0;
        nvencParams.enableSubFrameWrite = 0;
        nvencPresetConf.presetCfg.frameIntervalP = 1;
        nvencParams.encodeConfig = &nvencPresetConf.presetCfg;
        nvencParams.encodeConfig->gopLength = 1;
        nvencParams.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;

        winrt::check_hresult(NVENCStatHr(nvenc.nvEncInitializeEncoder(nvencPtr, &nvencParams)));

        cuCtxPushCurrent(nvencCtx);

        nvencFrame = new CUdeviceptr[nvencDesc.num_in_back_buffers];
        nvencRegisteredPtr = new NV_ENC_REGISTERED_PTR[nvencDesc.num_in_back_buffers];
        nvencMappedResource = new NV_ENC_INPUT_PTR[nvencDesc.num_in_back_buffers];
        nvencBistreamOutPtr = new NV_ENC_OUTPUT_PTR[nvencDesc.num_out_back_buffers];
        nvencCompletedFrame = new HANDLE[nvencDesc.num_out_back_buffers];

        for (int i = 0; i < nvencDesc.num_in_back_buffers; i++) {
            cuMemAllocPitch(&nvencFrame[i], &nvencFramePitch, nvencDesc.encoded_width * 4, nvencDesc.encoded_height, 16);

            NV_ENC_REGISTER_RESOURCE nv_enc_resource = { NV_ENC_REGISTER_RESOURCE_VER };
            nv_enc_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
            nv_enc_resource.bufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
            nv_enc_resource.resourceToRegister = (void*)nvencFrame[i];
            nv_enc_resource.width = nvencDesc.encoded_width;
            nv_enc_resource.height = nvencDesc.encoded_height;
            nv_enc_resource.pitch = nvencFramePitch;
            winrt::check_hresult(NVENCStatHr(nvenc.nvEncRegisterResource(nvencPtr, &nv_enc_resource)));
            nvencRegisteredPtr[i] = nv_enc_resource.registeredResource;

            NV_ENC_MAP_INPUT_RESOURCE nv_enc_map_res = { NV_ENC_MAP_INPUT_RESOURCE_VER };
            nv_enc_map_res.registeredResource = nvencRegisteredPtr[i];
            winrt::check_hresult(NVENCStatHr(nvenc.nvEncMapInputResource(nvencPtr, &nv_enc_map_res)));
            nvencMappedResource[i] = nv_enc_map_res.mappedResource;
        }

        for (int i = 0; i < nvDesc.num_out_back_buffers; i++) {
            NV_ENC_CREATE_BITSTREAM_BUFFER nv_enc_bitstream;
            nv_enc_bitstream = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
            winrt::check_hresult(NVENCStatHr(nvenc.nvEncCreateBitstreamBuffer(nvencPtr, &nv_enc_bitstream)));
            nvencBistreamOutPtr[i] = nv_enc_bitstream.bitstreamBuffer;

            nvencCompletedFrame[i] = CreateEvent(nullptr, false, false, nullptr);
            winrt::check_pointer(nvencCompletedFrame[i]);
            NV_ENC_EVENT_PARAMS eventParams = { NV_ENC_EVENT_PARAMS_VER };
            eventParams.completionEvent = nvencCompletedFrame[i];
            winrt::check_hresult(NVENCStatHr(nvenc.nvEncRegisterAsyncEvent(nvencPtr, &eventParams)));
        }

        for (int i = 0; i < NVEC_INTERNAL_EVENTS_NUM; i++)
            nvencInternalEvents[i] = CreateEvent(nullptr, false, false, nullptr);

        cuCtxPopCurrent(nullptr);

    } catch (winrt::hresult_error& err) {
        return err.code();
    }

    return S_OK;
}
