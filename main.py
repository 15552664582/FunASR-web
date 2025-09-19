from fastapi import Form, FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from funasr import AutoModel
from typing import List, Optional
import io
import soundfile as sf
import numpy as np
import librosa
import uvicorn
import traceback


try:
    model = AutoModel(
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        model_revision="v2.0.4",
        disable_update=True,
        cache_dir="/models/funasr",
    )
except Exception as e:
    print(f"模型加载失败，无法启动服务: {str(e)}")
    exit(1)

app = FastAPI(title="FunASR Service")


def error_response(status: str, text: str = ""):
    """统一的响应格式创建函数"""
    response_data = {
        "status": status,
        "text": text
    }
    return response_data


@app.post("/funasr/v1/asr")
async def asr_endpoint(file: UploadFile = File(...), prompt: Optional[List[str]] = Form([])):
    try:
        try:
            audio_bytes = await file.read()
            if len(audio_bytes) == 0:
                return JSONResponse(content=error_response("error", "上传的音频文件为空"))
        except Exception as e:
            return JSONResponse(content=error_response("error", f"文件读取失败: {str(e)}"))

        try:
            # 用 soundfile 从二进制读取 wav 数据
            audio_buffer = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(audio_buffer, dtype='float32')

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # 重采样到 16kHz
            if samplerate != 16000:
                data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)

            # 确保格式
            data = data.astype("float32")
        except Exception as e:
            return JSONResponse(content=error_response("error", f"音频格式错误: {str(e)}"))

        try:
            result = model.generate(
                input=data,
                sentence_timestamp=True,
                merge_vad=True,
                prompt=prompt,
            )
        except Exception as e:
            return JSONResponse(content=error_response("error", f"语音识别错误: {str(e)}"))

        # 返回结果
        return JSONResponse(content={
            "status": "success",
            "text": result[0].get("text", ""),
            "timestamp": result[0].get("timestamp", ""),
            "sentence": result[0].get("sentence_info", []),
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content=error_response("error", f"服务器内部错误: {str(e)}")
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

