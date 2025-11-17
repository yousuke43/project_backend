import httpx
import json
import os
from typing import Optional, Any

# --- VOICEVOX APIの設定 ---
# 環境変数からURLを読み込むことで、Docker環境に対応しやすくする
VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://host.docker.internal:50021")
# ---

async def get_supported_speakers() -> Optional[list[dict[str, Any]]]:
    """
    【非同期】利用可能な話者（スピーカー）のリストを取得する。
    """
    speakers_url = f"{VOICEVOX_URL}/speakers"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(speakers_url, timeout=10.0)
            response.raise_for_status()
            # キャラクター名とIDを一覧表示する
            speakers = response.json()
            print("--- 利用可能なキャラクター ---")
            for speaker in speakers:
                for style in speaker['styles']:
                    print(f"- {speaker['name']} ({style['name']}): speaker_id = {style['id']}")
            print("--------------------------")
            return speakers
    except httpx.RequestError as e:
        print(f"警告: スピーカー情報の取得に失敗しました。VOICEVOXエンジンは起動していますか？: {e}")
        return None

async def synthesize_voice(
    text: str, 
    speaker_id: int,
    speed_scale: float = 1.0,
    pitch_scale: float = 0.02,
    intonation_scale: float = 1.2,
    volume_scale: float = 1.0
) -> Optional[bytes]:
    """
    【非同期】テキストと話者IDでVOICEVOX APIからWAV形式の音声データを取得する。
    話速、ピッチ、抑揚、音量の調整機能を追加。
    
    Args:
        text (str): 読み上げたいテキスト。
        speaker_id (int): 使用する話者のID。
        speed_scale (float, optional): 話速。 Defaults to 1.3.
        pitch_scale (float, optional): ピッチ（声の高さ）。 Defaults to 0.02.
        intonation_scale (float, optional): 抑揚。 Defaults to 1.2.
        volume_scale (float, optional): 音量。 Defaults to 1.0.

    Returns:
        Optional[bytes]: 生成された音声データ（WAV形式のbytes）。失敗した場合はNone。
    """
    
    try:
        async with httpx.AsyncClient() as client:
            # 1. 音声合成クエリの作成 (audio_query)
            query_url = f"{VOICEVOX_URL}/audio_query"
            query_params = {"text": text, "speaker": speaker_id}
            
            query_response = await client.post(query_url, params=query_params, timeout=10.0)
            query_response.raise_for_status()
            audio_query = query_response.json()
            
            # 2. パラメータを音声合成クエリに上書き
            audio_query['speedScale'] = speed_scale
            audio_query['pitchScale'] = pitch_scale
            audio_query['intonationScale'] = intonation_scale
            audio_query['volumeScale'] = volume_scale
            
            # 3. 音声の生成 (synthesis)
            synthesis_url = f"{VOICEVOX_URL}/synthesis"
            synthesis_params = {"speaker": speaker_id}
            
            synthesis_response = await client.post(
                synthesis_url, 
                params=synthesis_params, 
                json=audio_query,
                timeout=20.0
            )
            synthesis_response.raise_for_status()
            
            return synthesis_response.content

    except httpx.HTTPStatusError as e:
        print(f"エラー: VOICEVOX APIでエラーが発生しました: Status {e.response.status_code}, Response: {e.response.text}")
        return None
    except httpx.RequestError as e:
        print(f"エラー: VOICEVOXエンジンへの接続に失敗しました。URLは正しいですか？ ({VOICEVOX_URL}): {e}")
        return None


