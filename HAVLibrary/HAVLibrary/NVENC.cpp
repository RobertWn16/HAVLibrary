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
    fclose(output_test);
}
winrt::hresult NVENC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    return winrt::hresult();
}

winrt::hresult NVENC::Encode(IFrame* in, unsigned char* buf, unsigned int& size)
{
    NVFrame* nv_frame = dynamic_cast<NVFrame*>(in);
    if (nv_frame)
    {
        NVENCSTATUS stat = NV_ENC_SUCCESS;

        cuCtxPushCurrent(deviceCtx);

        CUDA_MEMCPY2D m = { 0 };
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        m.srcDevice = nv_frame->cuFrame;
        m.srcPitch = pitch;
        m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        m.dstDevice = frame;
        m.dstPitch = pitch;
        m.WidthInBytes = m.dstPitch;
        m.Height = 2160;
        CUresult res = cuMemcpy2D(&m);
 

        NV_ENC_MAP_INPUT_RESOURCE nv_enc_map_res = { NV_ENC_MAP_INPUT_RESOURCE_VER };
        nv_enc_map_res.registeredResource = nv_enc_res;
        stat = functionList.nvEncMapInputResource(nvencEncoder, &nv_enc_map_res);

        NV_ENC_PIC_PARAMS pic_params = { NV_ENC_PIC_PARAMS_VER };
        pic_params.version = NV_ENC_PIC_PARAMS_VER;
        pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
        pic_params.inputBuffer = nv_enc_map_res.mappedResource;
        pic_params.inputHeight = 2160;
        pic_params.inputWidth = 3840;
        pic_params.inputPitch = 0;
        pic_params.outputBitstream = nv_enc_bitstream.bitstreamBuffer;
        
        stat = functionList.nvEncEncodePicture(nvencEncoder, &pic_params);
        NV_ENC_LOCK_BITSTREAM bitsream_lock = { NV_ENC_LOCK_BITSTREAM_VER };
        bitsream_lock.outputBitstream = nv_enc_bitstream.bitstreamBuffer;
        bitsream_lock.doNotWait = false;
        stat = functionList.nvEncLockBitstream(nvencEncoder, &bitsream_lock);
        size = bitsream_lock.bitstreamSizeInBytes;
        fwrite(bitsream_lock.bitstreamBufferPtr, 1, size, output_test);
        stat = functionList.nvEncUnlockBitstream(nvencEncoder, nv_enc_bitstream.bitstreamBuffer);

        stat = functionList.nvEncUnmapInputResource(nvencEncoder, &nv_enc_map_res);

        cuCtxPushCurrent(nullptr);
        //std::cout << "Is ok " << stat;
    }
    return S_OK;
}

winrt::hresult NVENC::ConfigureEncoder(CUcontext deviceContext)
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
        NV_ENC_BUFFER_FORMAT format[30];
        NV_ENC_CONFIG nvencConf;
        uint32_t inputFormatCount;
        NV_ENC_INITIALIZE_PARAMS nvencParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        uint32_t numOfPreset;

        memset(&nvencParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
        winrt::check_pointer(&deviceContext);
        winrt::check_hresult(NVENCStatHr(NvEncodeAPICreateInstance(&functionList)));
        nvencOpenPar.apiVersion = NVENCAPI_VERSION;
        nvencOpenPar.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        nvencOpenPar.device = deviceContext;
        winrt::check_hresult(NVENCStatHr(functionList.nvEncOpenEncodeSessionEx(&nvencOpenPar, &nvencEncoder)));

        functionList.nvEncGetEncodeGUIDCount(nvencEncoder, &numOfGuids);
        functionList.nvEncGetEncodeGUIDs(nvencEncoder, guids, 5, &numOfGuids);

        stat = functionList.nvEncGetEncodePresetCount(nvencEncoder, guids[10], &numOfPreset);
        stat = functionList.nvEncGetEncodePresetGUIDs(nvencEncoder, guids[0], presetGuids, 20, &numOfPreset);
        stat = functionList.nvEncGetEncodePresetConfigEx(nvencEncoder, guids[0], presetGuids[0], NV_ENC_TUNING_INFO_LOSSLESS, &nvencPresetConf);
        stat = functionList.nvEncGetInputFormatCount(nvencEncoder, guids[0], &inputFormatCount);
        stat = functionList.nvEncGetInputFormats(nvencEncoder, guids[0], format, 30, &inputFormatCount);

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
        stat = functionList.nvEncInitializeEncoder(nvencEncoder, &nvencParams);

        deviceCtx = deviceContext;
        cuCtxPushCurrent(deviceCtx);


        CUresult error = cuMemAllocPitch(&frame, &pitch, 3840 * 4, 2160, 16);
        NV_ENC_REGISTER_RESOURCE nv_enc_resource = { NV_ENC_REGISTER_RESOURCE_VER };
        nv_enc_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        nv_enc_resource.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
        nv_enc_resource.resourceToRegister = (void*)frame;
        nv_enc_resource.width = 3840;
        nv_enc_resource.height = 2160;
        nv_enc_resource.pitch = pitch;

        stat = functionList.nvEncRegisterResource(nvencEncoder, &nv_enc_resource);
        nv_enc_res = nv_enc_resource.registeredResource;

        fopen_s(&output_test, "output.h264", "wb");


        nv_enc_bitstream = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
        stat = functionList.nvEncCreateBitstreamBuffer(nvencEncoder, &nv_enc_bitstream);
        HANDLE completion = CreateEvent(NULL, FALSE, FALSE, NULL);
        NV_ENC_EVENT_PARAMS eventParams = { NV_ENC_EVENT_PARAMS_VER };
        eventParams.completionEvent = completion;
        stat = functionList.nvEncRegisterAsyncEvent(nvencEncoder, &eventParams);
        cuCtxPopCurrent(nullptr);


        }catch (winrt::hresult_error& err) {
        return err.code();
    }
    return S_OK;
}
