import asyncio
import json
import os
from datetime import date, datetime
from typing import Optional, Any

import httpx
import numpy as np
import torch
import csv
import requests  # requestsは未使用ですが、元のimportリストに残しています
import re
import uvicorn
from faster_whisper import WhisperModel
import traceback # ★ traceback をインポート (エラー表示用)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException # ★ HTTPException を追加
from fastapi.middleware.cors import CORSMiddleware
import voicevox_util
from pathlib import Path
from pydantic import BaseModel

from collections import deque
import numpy as np
from pydantic import BaseModel
from typing import Optional, Any # 既存のimportに追加

# --- グローバル設定 (Global Settings) ---
DATA_DIR = "./data"
DATA_FILE = os.path.join(DATA_DIR, "data.json")
MEMORY_FILE = os.path.join(DATA_DIR, "Memory.csv")
HEALTH_FILE = os.path.join(DATA_DIR,"Health.csv")
EEG_LOG_FILE = os.path.join(DATA_DIR, "eeg_events_log.jsonl")
TRAITS_FILE = os.path.join(DATA_DIR,"Traits.csv")
today_str = date.today().isoformat()



# Difyから取得したAPIキーとURLを環境変数から読み込む
# Load Dify API key and URLs from environment variables
API_KEY = os.getenv("DIFY_API_KEY", "YOUR_DIFY_API_KEY")
# 通常のチャット用エンドポイント
CHAT_API_URL = os.getenv("DIFY_CHAT_URL", "http://host.docker.internal/v1/chat-messages")
# 思い出登録用ワークフローのエンドポイント
REGISTER_WORKFLOW_URL = os.getenv("DIFY_REGISTER_URL", "http://host.docker.internal/v1/chat-messages")

# DifyのナレッジベースID (管理画面のURLなどから確認)
DATASET_ID = os.getenv("DIFY_DATASET_ID", "YOUR_ACTUAL_DATASET_ID")
DIFY_DATASETS_API_KEY = os.getenv("DIFY_DATASETS_API_KEY", "YOUR_DIFY_API_KEY")

DATASET_URL = f"http://host.docker.internal/v1/datasets/{DATASET_ID}/document/create-by-file"

SPEAKER_ID = 20  
OUTPUT_FILENAME = "generated_voice.wav"  # 音声データの保存ファイル名は、今回は使用しない（WebSocketで直接送信するため）

print(f"--- 読み込まれたキーの確認: '{API_KEY}' ---")

# --- 1. FastAPI インスタンス作成 (Create FastAPI instance) ---
app = FastAPI()
print("FastAPI サーバーを初期化しました。")


# ★ 3. ミドルウェアをアプリに追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 全てのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],  # 全てのHTTPメソッドを許可（GET, POST, PUT, DELETE, etc）
    allow_headers=["*"],  # 全てのヘッダーを許可
)
# --- 2. Whisperモデルロード (Load Whisper model) ---
# GPUが利用可能かチェック
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model_size = "large-v3"

print(f"Whisperモデルをロード中... (デバイス: {device}, 計算タイプ: {compute_type}, モデル: {model_size})")
try:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("Whisperモデルのロード完了。")
except Exception as e:
    print(f"Whisperモデルのロードに失敗しました: {e}")
    exit()

# --- 3. Silero VADモデルロード (Load Silero VAD model) ---
print("Silero VADモデルをロード中...")
try:
    # utilsを明示的に取得
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    print("Silero VADモデルのロード完了。")
except Exception as e:
    print(f"Silero VADモデルのロードに失敗しました: {e}")
    vad_model = None
    utils = None


# ★★★ 1. 脳波の状態を管理するクラスを定義 ★★★
# gpu-transcriber-service.py の EEGState クラスを修正

# --- ★★★ 2. 最新の室内イベントを保持するグローバル変数 ★★★ ---
latest_indoor_event: Optional[dict] = None
event_lock = asyncio.Lock() # 非同期処理で安全にアクセスするためのロック


class EEGEvent(BaseModel):
    timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_name: str
    event_type: str
    arousal_value: float

# ★★★ 3. 室内イベント受け取り用の Pydantic モデルを追加 ★★★
class IndoorEEGEvent(BaseModel):
    timestamp: str
    event_type: str # "focus_sustained", "relax_spike", "arousal_spike" など

# --- 脳波サマリー用のヘルパー関数 ---
def format_event_to_sentence(event_data: dict) -> str:
    """脳波イベントの辞書データを自然な日本語の文章に変換する"""
    try:
        time_str = datetime.fromisoformat(event_data["timestamp"]).strftime("%H時%M分頃")
        place_name = event_data.get("place_name", "不明な場所")
        return f"・{time_str}、{place_name}で、何かに強く興味を惹かれたようです。"
    except: return ""

async def get_eeg_summary() -> Optional[str]:
    """今日の脳波イベントログを読み込み、LLM用の要約テキストを作成する"""
    if not os.path.exists(EEG_LOG_FILE): 
        print("脳波ログファイルが見つかりません。")
        return None
    
    today_events = []
    try:
        with open(EEG_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if "arousal_value" in event and datetime.fromisoformat(event["timestamp"]).date() == date.today():
                        today_events.append(event)
                except: continue
        
        if not today_events: 
            print("今日の脳波イベントはありません。")
            return None

        summary = "\n".join(filter(None, [format_event_to_sentence(e) for e in today_events]))
        highlight = max(today_events, key=lambda e: e.get("arousal_value", 0))
        summary += f"\nこの中で特に反応が強かったのは、{highlight.get('place_name', 'ある場所')}での出来事のようです。"
        print(f"脳波サマリーを作成しました:\n{summary}")
        return summary
    except Exception as e:
        print(f"🚨 脳波サマリー作成中にエラー: {e}")
        return None

class EEGRawData(BaseModel):
    timestamp: str
    focus: float
    relax: float
    arousal: float

# gpu-transcriber-service.py の修正箇所
@app.get("/get_health_data")
async def get_health_data_csv():
    """
    Health.csv ファイルの内容をJSON配列として返します。
    React (Chart.js) が期待する数値型に変換します。
    """
    if not os.path.exists(HEALTH_FILE):
        print(f"🚨 API /get_health_data: {HEALTH_FILE} が見つかりません。")
        raise HTTPException(status_code=404, detail=f"{os.path.basename(HEALTH_FILE)} not found")
    
    health_data_list = []
    try:
        # 'utf-8-sig' でBOM (Excelなどが付ける不可視の文字) を処理
        with open(HEALTH_FILE, mode='r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            
            print(f"✅ API /get_health_data: CSVヘッダー読み込み: {reader.fieldnames}")

            for row in reader:
                try:
                    # React側が数値を期待するため、型変換を行う
                    # React側のコード (HealthPage.jsx) が期待するキー名に合わせる
                    processed_row = {
                        "date": row.get('date'),
                        "体重": float(row.get('体重')),
                        "歩数": int(row.get('歩数')),
                        "睡眠時間": float(row.get('睡眠時間')),
                        "最高血圧": int(row.get('最高血圧')),
                        "最低血圧": int(row.get('最低血圧')),
                        "消費カロリー": int(row.get('消費カロリー'))
                    }
                    health_data_list.append(processed_row)
                except (ValueError, TypeError, KeyError) as convert_error:
                    # データが空 (None) だったり、数値に変換できない、またはキーが存在しない場合はその行をスキップ
                    print(f"⚠️ API /get_health_data: 行をスキップ (型変換/キーエラー): {row} - {convert_error}")
                    continue
        
        print(f"✅ API /get_health_data: {len(health_data_list)} 件の健康データをJSONで送信します。")
        return health_data_list
        
    except Exception as e:
        print(f"🚨 API /get_health_data: CSV読み込み中にエラー: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reading health CSV: {e}")

@app.get("/get_memories")
async def get_memories_csv():
    """
    Memory.csv ファイルの内容をJSON配列として返します。
    (BOMを処理し、キー名をReactが期待する形に正規化します)
    """
    if not os.path.exists(MEMORY_FILE):
        print(f"🚨 API /get_memories: {MEMORY_FILE} が見つかりません。")
        raise HTTPException(status_code=404, detail=f"{os.path.basename(MEMORY_FILE)} not found")
    
    memories_normalized = []
    try:
        # ★ encoding='utf-8-sig' でBOMを自動的に処理
        with open(MEMORY_FILE, mode='r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            
            print(f"✅ API /get_memories: CSVヘッダー読み込み: {reader.fieldnames}")

            for row in reader:
                # ★ React側が期待するキー名 ("日付", "トピック", "内容") にマッピング
                normalized_row = {
                    "日付": row.get("日付"), # utf-8-sigでBOMが除去された "日付" キー
                    # "タイトル" キーか "トピック" キーのどちらかに対応し、"トピック" に統一
                    "トピック": row.get("タイトル") or row.get("トピック"), 
                    "内容": row.get("内容")
                }
                
                # (念のため) BOM除去がうまくいかなかった場合
                if normalized_row["日付"] is None:
                    normalized_row["日付"] = row.get("﻿日付") # BOM付きキーを試す
                
                memories_normalized.append(normalized_row)
        
        print(f"✅ API /get_memories: {len(memories_normalized)} 件の思い出を正規化して送信します。")
        # ★ 正規化済みのリストを返す
        return memories_normalized
    except Exception as e:
        print(f"🚨 API /get_memories: CSV読み込み中にエラー: {e}")
        traceback.print_exc() # サーバーログに詳細なエラーを表示
        raise HTTPException(status_code=500, detail=f"Error reading memory CSV: {e}")

# ( ... @app.post("/log_event") ... はそのまま ... )}")

@app.post("/log_event")
async def log_eeg_event(event: EEGEvent):
    """
    クライアントから脳波イベントデータを受け取り、
    JSON Lines形式 (.jsonl) のファイルに追記して保存するエンドポイント。
    """
    print(f"📡 イベント受信: {event.place_name} (覚醒度: {event.arousal_value:.2f})")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(EEG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        print(f"💾 イベントを '{EEG_LOG_FILE}' に保存しました。")
        return {"status": "success"}
    except Exception as e:
        print(f"🚨 イベントのファイル保存中にエラーが発生しました: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/log_indoor_event")
async def log_indoor_eeg_event(event: IndoorEEGEvent):
    """
    在宅用クライアントから検知された脳波イベント (集中など) を受け取るエンドポイント
    """
    global latest_indoor_event
    event_type = event.event_type
    print(f"🏠 室内イベント受信: '{event_type}'")
    try:
        async with event_lock: # グローバル変数を安全に更新
            latest_indoor_event = event.model_dump() # 辞書として保存
        print(f"💾 最新の室内イベントを '{event_type}' に更新しました。")
        return {"status": "success"}
    except Exception as e:
        print(f"🚨 室内イベントの保存中にエラー: {e}")
        # クライアントにエラーを返す (500 Internal Server Error)
        raise HTTPException(status_code=500, detail=f"Error saving indoor event: {e}")
    
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("クライアントが接続しました！")

    if not vad_model:
        print("VADモデルが利用できません。接続を閉じます。")
        await websocket.close(code=1011, reason="VAD model is not available")
        return

    # --- WebSocket接続ごとの状態管理 ---
    vad_iterator = VADIterator(vad_model, threshold=0.5)
    audio_buffer = bytearray()
    llm_wating = False
    today_check = False
    conversation_id = None

    # 会話履歴を保存するためのリストを初期化
    chat_history: list[dict[str, str]] = []

    traits_file_content = ""

    try:
        # ファイルを読み込みモード('r')で開く
        # encoding='utf-8' を指定して日本語の文字化けを防ぐ
        with open(TRAITS_FILE, 'r', encoding='utf-8') as f:
            # .read() でファイルの内容すべてを文字列として読み込む
            traits_file_content = f.read()
        
        # 読み込んだ内容の確認 (任意)
        print(f"--- {TRAITS_FILE} の内容を読み込みました ---")
        print(traits_file_content)
        print("-----------------------------------")

    except FileNotFoundError:
        print(f"🚨 エラー: {TRAITS_FILE} が見つかりません。")
        # traits_file_content は空文字列 "" のままになります

    except Exception as e:
        print(f"🚨 ファイル読み込み中に予期せぬエラーが発生しました: {e}")
    # traits_file_content は空文字列 "" のままになります

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            conversation_id = data.get("conversation_id")
            print(f"前回の会話IDを読み込みました: {conversation_id}")
    except (FileNotFoundError, json.JSONDecodeError):
        print("データファイルが見つからないか空です。新しい会話から開始します。")
        data = {}

    # --- 内部関数定義 ---

    

    async def checkLastDate():
        nonlocal today_check
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_data = {}

        last_date = current_data.get("last_conversation_date")
        print(last_date)
        if last_date != today_str:
            print(f"今日の日付：{today_str}")
            print(f"前回の会話日は {last_date}。今日の 脳波 データを入手します。")
            today_check = True
            current_data["last_conversation_date"] = today_str
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)
            print("last_conversation_date を更新しました。")
        else:
            print("今日すでに 脳波 データは処理済みです。")

    async def sendToLLM(message: str):
        nonlocal llm_wating, today_check, conversation_id, chat_history
        global latest_indoor_event # ★ グローバル変数を参照

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data_payload = {
            "inputs": {
                "mode": "talk",
                "current_data": today_str,
                "personality_traits":traits_file_content,
                "server_trigger":""
            },
            "query": message,
            "user": "docker-user-001",
            "response_mode": "blocking"
        }

        if conversation_id:
            data_payload["conversation_id"] = conversation_id

        try:
            if today_check:
                eeg_data = await get_eeg_summary()
                data_payload['inputs']['eeg_summary'] = eeg_data
                print(f"LLMにメッセージと脳波位置情報データを送信 (Blocking): {message} {eeg_data}")
                today_check = False
            else:
                
                print(f"LLMにメッセージを送信 (Blocking): {message}")

            event_to_send = None
            processing_data = {"type": "ai_processing", "text": "（考え中...）"}
            await websocket.send_text(json.dumps(processing_data, ensure_ascii=False))
            async with event_lock: # 安全に読み取り＆リセット
                if latest_indoor_event:
                    event_to_send = latest_indoor_event.get("event_type")
                    latest_indoor_event = None # ★ 送信したらリセット (消費)

            if event_to_send:
                data_payload['inputs']['server_trigger'] = event_to_send
                print(f"🔥 Difyペイロードに[在宅トリガー] ({event_to_send}) を追加しました。")

            print(f"Difyに送信するペイロード: {json.dumps(data_payload, indent=2, ensure_ascii=False)}")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(CHAT_API_URL, headers=headers, json=data_payload)
                response.raise_for_status()
                json_data = response.json()
                print(data_payload)

                final_answer = json_data.get("answer", "[エラー: 応答を取得できませんでした]")

                if "応答生成:" in final_answer:
                    # "応答生成:" より後の部分を抽出
                    parts = final_answer.split("応答生成:")
                    final_answer = parts[1]
                else:
                    # 区切り文字がなければ、全体を応答として扱う
                    pass
                new_conv_id = json_data.get("conversation_id")

                if new_conv_id:
                    conversation_id = new_conv_id

                # 会話のやり取りを履歴リストに追加
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "ai", "content": final_answer})
                print("会話履歴に今回のやり取りを追加しました。")

                print(f"Difyからの最終応答: {final_answer}")
                print(f"Conversation ID: {conversation_id}")

                # ★★★ 修正された音声合成と応答ロジック ★★★
                wav_data = await voicevox_util.synthesize_voice(final_answer, SPEAKER_ID)

                if wav_data:
                    # 音声合成成功: WAVファイルをクライアントに送信 (send_bytesを使用)
                    print(f"\n✅ 完了: 音声データ ({len(wav_data)} バイト) をクライアントに送信します。")

                    data = {
                        "type": "ai_response",
                        "text": final_answer
                    }
                    # 辞書をJSON形式の文字列に変換
                    # ensure_ascii=False は日本語を正しく扱うために重要です
                    json_string = json.dumps(data, ensure_ascii=False)

                    # 文字列として送信
                    await websocket.send_text(json_string)
                    # クライアントへの応答として、生成された音声データ (WAV) を送信
                    await websocket.send_bytes(wav_data)

                else:
                    # 音声合成失敗: フォールバックとしてテキストをクライアントに送信
                    print("\n❌ 音声合成に失敗しました。VOICEVOXエンジンが起動しているか確認してください。テキストを代替応答として送信します。")
                    await websocket.send_text(final_answer)  # テキストをクライアントに送信
                # ★★★ 修正終わり ★★★

        except httpx.HTTPStatusError as e:
            error_message = f"[エラー] Dify APIエラー: {e.response.status_code}, {e.response.text}"
            print(error_message, API_KEY)
            await websocket.send_text(error_message)
        except Exception as e:
            error_message = f"[エラー] sendToLLMで予期せぬエラー: {e}"
            print(error_message)
            await websocket.send_text(error_message)
        finally:
            llm_wating = False
            print("LLM応答待ちフラグをリセット。")

    async def getNewMemory(history: list[dict[str, str]]):
        if not history:
            print("会話履歴が存在しないため、思い出の抽出をスキップします。")
            return None

        # 会話履歴リストをDifyが読みやすい単一の文字列に変換
        formatted_history = ""
        for turn in history:
            if turn["role"] == "user":
                formatted_history += f"ユーザー: {turn['content']}\n"
            else:
                formatted_history += f"AI: {turn['content']}\n"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # レジスターモードのワークフローに渡すデータを作成
        data_payload = {
            "inputs": {
                # Difyワークフローの開始ノードで定義した変数名に合わせる
                "chat_history": formatted_history,
                "mode": "register"
            },
            "query": "test",
            "user": "docker-user-001",
            "response_mode": "blocking"
        }

        print("会話の要約と新しい思い出の抽出をDifyにリクエストします...")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # レジスターモード専用のURLを呼び出す
                response = await client.post(REGISTER_WORKFLOW_URL, headers=headers, json=data_payload)
                response.raise_for_status()
                json_data = response.json()

                extracted_memory = json_data.get("answer")

                if extracted_memory:
            
                    separator_pattern = re.compile(r"^(traits?|特性)\s*:", re.IGNORECASE | re.MULTILINE)
                    match = separator_pattern.search(extracted_memory)
                    
                    # --- 変数をここで初期化 ---
                    memories_part = ""
                    traits_part = ""

                    if match:
                        # --- 1. 区切り文字が見つかった場合 ---
                        memories_part = extracted_memory[:match.start()].strip()
                        traits_part = extracted_memory[match.end():].strip()
                        print("区切り文字が見つかり、思い出と特性に分割しました。")
                    else:
                        # --- 2. 区切り文字が見つからなかった場合 (elseブロックの追加) ---
                        memories_part = extracted_memory.strip()
                        # traits_part は空のまま
                        print("区切り文字が見つからず、全体を思い出として処理します。")

                    print("--- 抽出された思い出 (保存対象) ---")
                    print(memories_part)
                    print("-----------------------------")
                    print("--- 抽出された特性 (保存対象) ---")
                    print(traits_part)
                    print("-----------------------------")

                    # --- 3. 正しい関数を呼び出す (修正点) ---
                    
                    # (1) 思い出を保存
                    if memories_part:
                        await save_memories_to_csv(memories_part)
                    
                    # (2) 特性を保存
                    if traits_part:
                        await save_traits_to_csv(traits_part)

                    # 関数としては抽出した思い出部分を返す (これは元の設計と同じ)
                    return memories_part
            
                else: # (これは元のコードの else)
                    print("思い出は抽出されませんでした（応答が空でした）。")
                    return None

        except httpx.HTTPStatusError as e:
            print(f"[エラー] 思い出抽出APIエラー: {e.response.status_code}, {e.response.text}")
            return None
        except Exception as e:
            print(f"[エラー] 思い出抽出中に予期せぬエラー: {e}")
            return None

    async def save_memories_to_csv(memories_string: str):
        if not memories_string or not isinstance(memories_string, str):
            print("保存する新しい思い出がありません。")
            return

        today_str = date.today().isoformat()
        new_rows = []

        # CSV解析部分
        memory_list = memories_string.strip().split('\n')
        for memory_line in memory_list:
            try:
                # 正規表現で最初のコロンまたは全角コロンで分割
                parts = re.split(r'[:：]', memory_line, 1)
                if len(parts) == 2:
                    title = parts[0].strip()
                    content = parts[1].strip()
                    new_rows.append([today_str, title, content])
                elif memory_line.strip():
                    new_rows.append([today_str, "その他", memory_line.strip()])
            except Exception as e:
                print(f"Error parsing memory line: '{memory_line}', Error: {e}")
                continue

        if not new_rows:
            print("解析の結果、保存する新しい思い出がありませんでした。")
            return

        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            file_exists = os.path.isfile(MEMORY_FILE)
            with open(MEMORY_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["日付", "トピック", "内容"])
                writer.writerows(new_rows)
            print(f"{len(new_rows)}件の新しい思い出をCSVに保存しました。")

            # --- ここからがAPIリクエスト部分 (Difyナレッジ更新) ---
            if not DATASET_ID or DATASET_ID == "YOUR_ACTUAL_DATASET_ID":
                print("DIFY_DATASET_IDが設定されていないため、ナレッジの更新をスキップします。")
                return

            endpoint = DATASET_URL
            headers = {
                "Authorization": f"Bearer {DIFY_DATASETS_API_KEY}"
            }
            # ファイルを開いて準備
            with open(MEMORY_FILE, 'rb') as mem_file:
                files = {
                    'file': (MEMORY_FILE.split('/')[-1], mem_file, 'text/csv')
                }
                process_rule = {"mode": "automatic"}
                data = {
                    "process_rule": json.dumps(process_rule)
                }

                print("Difyナレッジへのファイルアップロードを開始します...")

                # httpx の非同期処理に置き換え
                async with httpx.AsyncClient(timeout=60.0) as client:
                    try:
                        response = await client.post(endpoint, headers=headers, files=files, data=data)
                        response.raise_for_status()
                        print("ファイルのアップロードに成功しました。")
                        print("レスポンス:", response.json())
                    except httpx.HTTPStatusError as e:
                        print(f"[エラー] Difyナレッジ更新APIエラー: {e.response.status_code}, {e.response.text}")
                    except httpx.RequestError as e:
                        print(f"[エラー] Difyナレッジ更新中にリクエストエラーが発生しました: {e}")

        except Exception as e:
            print(f"CSVファイルへの書き込みまたはアップロード中にエラーが発生しました: {e}")

    async def save_traits_to_csv(traits_string: str):

        if not traits_string or not isinstance(traits_string, str):
            print("保存する特性がありません。")
            return

        today_str_csv = date.today().isoformat()
        new_rows = []

        # 文字列を改行でリスト化
        trait_list = traits_string.strip().split('\n')
        
        for trait_line in trait_list:
            trait_line = trait_line.strip()
            if not trait_line: 
                continue # 空行はスキップ

            # 1. 箇条書きマーク ( *, - ) があれば除去
            if trait_line.startswith(('*', '-')):
                trait_line = trait_line[1:].strip()

            # 2. プレフィックス (特性:, Trait:) があれば除去
            # re.sub を使って、行頭の "特性:" や "Trait:" (大文字小文字無視) を空文字列に置換
            prefix_pattern = re.compile(r"^(特性|Trait)\s*[:：]\s*")
            trait_content = prefix_pattern.sub("", trait_line).strip()

            # 3. 内容が残っていればリストに追加
            if trait_content:
                new_rows.append([today_str_csv, trait_content])

        if not new_rows:
            print("解析の結果、保存する新しい特性がありませんでした。")
            return

        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            # 保存先ファイル (TRAITS_FILE) の存在チェック
            file_exists = os.path.isfile(TRAITS_FILE)
            
            with open(TRAITS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 1. ファイルが存在しないか、中身が空の場合のみヘッダーを書き込む
                if not file_exists or os.path.getsize(TRAITS_FILE) == 0:
                    writer.writerow(["日付", "特性"]) # ヘッダー
                    
                # 2. 新しい行（特性）を追記
                writer.writerows(new_rows)
                
            print(f"{len(new_rows)}件の特性を {TRAITS_FILE} に保存しました。")

        except Exception as e:
            print(f"🚨 特性CSVファイル ({TRAITS_FILE}) への書き込み中にエラー: {e}")


    # --- メイン処理開始 ---
    await checkLastDate()

    try:
        while True:
            # WebSocketからデータを受け取る
            message = await websocket.receive()
            if llm_wating: continue

            # 1. テキストが送られてきた場合の処理
            if "text" in message:
                received_text = message["text"]
                print(f"💬 テキストメッセージ受信: '{received_text}'")
                
                # 文字起こしをスキップして、直接AIに送る
                llm_wating = True
                asyncio.create_task(sendToLLM(received_text))

            # VAD処理
            elif "bytes" in message:
                data_bytes = message["bytes"]
                audio_buffer.extend(data_bytes)
                audio_int16 = np.frombuffer(data_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0

                speech_dict = vad_iterator(torch.from_numpy(audio_float32), return_seconds=True)

                if speech_dict and 'end' in speech_dict:
                    print("発話終了を検出。文字起こしを実行します...")
                    llm_wating = True
                    await asyncio.sleep(0.3)  # わずかな遅延を挿入

                    full_audio_float32 = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                    # FasterWhisperで文字起こし
                    segments, _ = model.transcribe(
                        full_audio_float32,
                        beam_size=5,
                        language="ja",
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )
                    transcription = "".join([s.text for s in segments]).strip()
                    print(f"文字起こし結果: {transcription}")

                    audio_buffer.clear()
                    vad_iterator.reset_states()
                    print("--- バッファリセット完了、LLM処理を開始 ---")

                    if transcription:
                        # LLMへの送信は非同期タスクとして実行
                        
                        # 送りたいデータを作成 (辞書型)
                        data = {
                            "type": "user_transcription",
                            "text": transcription
                        }

                        # 辞書をJSON形式の文字列に変換
                        # ensure_ascii=False は日本語を正しく扱うために重要です
                        json_string = json.dumps(data, ensure_ascii=False)

                        # 文字列として送信
                        await websocket.send_text(json_string)
                        asyncio.create_task(sendToLLM(transcription))
                    else:
                        llm_wating = False
                        print("空の発話だったのでスキップ。")

    except WebSocketDisconnect:
        print("クライアントが切断しました。")
    except Exception as e:
        print(f"websocket_endpointでエラーが発生しました: {e}")
    finally:
        # 接続終了時に最新のconversation_idをファイルに保存
        if conversation_id:
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                file_data = {}

            file_data["conversation_id"] = conversation_id
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False, indent=4)
            print(f"最新の会話ID ({conversation_id}) をファイルに保存しました。")

        vad_iterator.reset_states()

        # ▼▼▼ テスト用：ここにサンプル会話履歴をハードコード ▼▼▼
        print("--- テストモード: サンプル会話履歴を使用して思い出を抽出します ---")
        test_chat_history = [
            {"role": "user", "content": "こんばんは。今日は少し肌寒いね。こういう日は、熱燗が恋しくなるよ。"},
            {"role": "ai", "content": "すっかり秋めいてきましたね。熱燗、いいですね。何か肴でもご用意しましょうか？"},
            {"role": "user", "content": "ありがとう。そうだな、イカの塩辛でもあれば嬉しいな。そういえば、高校生の頃、よく学校帰りに友達とラーメン屋に寄ったんだ。そこのおじさんが出してくれるお新香が絶品でね。"},
            {"role": "ai", "content": "放課後のラーメン、青春の味ですね。お友達とどんなお話をされていたのですか？"},
            {"role": "user", "content": "くだらない話ばかりだよ。部活のこととか、好きな音楽のこととか。でも、なぜかあのラーメン屋で話すと、将来の夢みたいな大きな話も素直にできたんだ。不思議なもんだね。"},
            {"role": "ai", "content": "特別な場所だったのですね。お友達とは今でもご連絡を？"},
            {"role": "user", "content": "ああ、今でも年に一度は集まって、あの頃の話をするよ。もちろん、あのラーメン屋の話もね。"}
        ]

        # 接続終了時に、テスト用の会話履歴を使って思い出を抽出
        await getNewMemory(chat_history)

        # 本番運用時は、上の行をコメントアウトし、下の行のコメントを解除します
        # await getNewMemory(chat_history)

        print("接続が終了しました。")


# --- 5. FastAPI を起動 ---
if __name__ == "__main__":
    if API_KEY == "YOUR_DIFY_API_KEY":
        print("\n[警告] DIFY_API_KEYが設定されていません。")
        print(".envファイルを作成し、DIFY_API_KEY='your_actual_api_key' のように設定してください。\n")
    # ファイル名がこのファイル自身であると仮定して、以下のように修正
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)