import asyncio
import json
import os
from datetime import date
from typing import Optional, Any

import httpx
import numpy as np
import torch
import csv
import requests # requestsは未使用ですが、元のimportリストに残しています
import re
import uvicorn
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import voicevox_util
from pathlib import Path

# --- グローバル設定 (Global Settings) ---
DATA_DIR = "./data"
DATA_FILE = os.path.join(DATA_DIR, "data.json")
MEMORY_FILE =os.path.join(DATA_DIR,"Memory.csv")
FITBIT_FILE=os.path.join(DATA_DIR, "Fitbit.csv")
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
DIFY_DATASETS_API_KEY= os.getenv("DIFY_DATASETS_API_KEY","YOUR_DIFY_API_KEY")

DATASET_URL = f"http://host.docker.internal/v1/datasets/{DATASET_ID}/document/create-by-file"

SPEAKER_ID = 3 # 例: 3 (春日部つむぎ ノーマル)
OUTPUT_FILENAME = "generated_voice.wav" # 音声データの保存ファイル名は、今回は使用しない（WebSocketで直接送信するため）

print(f"--- 読み込まれたキーの確認: '{API_KEY}' ---")

# --- 1. FastAPI インスタンス作成 (Create FastAPI instance) ---
app = FastAPI()
print("FastAPI サーバーを初期化しました。")

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


# --- 4. WebSocket エンドポイント (WebSocket Endpoint) ---
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
    fitbit_sending = False
    conversation_id = None
    
    # 会話履歴を保存するためのリストを初期化
    chat_history: list[dict[str, str]] = [] 

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

    async def getFitbitData():
        return {"steps": 12000, "sleep_hours": 7.5}

    async def checkLastDate():
        nonlocal fitbit_sending
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_data = {}
        
        last_date = current_data.get("last_conversation_date")
        if last_date != today_str:
            print(f"前回の会話日は {last_date}。今日の FitBit データを入手します。")
            fitbit_sending = True
            current_data["last_conversation_date"] = today_str
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)
            print("last_conversation_date を更新しました。")
        else:
            print("今日すでに FitBit データは処理済みです。")

    async def sendToLLM(message: str):
        nonlocal llm_wating, fitbit_sending, conversation_id, chat_history

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data_payload = {
            "inputs": {
                "mode": "talk"          
            },
            "query": message,
            "user": "docker-user-001",
            "response_mode": "blocking" 
        }

        if conversation_id:
            data_payload["conversation_id"] = conversation_id
        
        try:
            if fitbit_sending:
                fitbit_data = await getFitbitData()
                data_payload['inputs']['fitbit_context'] = f"ユーザーの活動データ: 歩数 {fitbit_data['steps']}歩, 睡眠時間 {fitbit_data['sleep_hours']}時間"
                print(f"LLMにメッセージとFitbitデータを送信 (Blocking): {message}")
                fitbit_sending = False
            else:
                print(f"LLMにメッセージを送信 (Blocking): {message}")

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(CHAT_API_URL, headers=headers, json=data_payload)
                response.raise_for_status()
                json_data = response.json()

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
                    
                    # デバッグ/確認用にファイル保存を行う場合:
                    # output_path = Path(OUTPUT_FILENAME)
                    # output_path.write_bytes(wav_data)
                    # print(f"デバッグ用: 音声は '{output_path.resolve()}' に保存されました。")
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
                    await websocket.send_text(final_answer) # テキストをクライアントに送信
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
                    print("新しい思い出の抽出に成功しました。")
                    print("--- 抽出された内容 ---")
                    print(extracted_memory)
                    print("----------------------")
                    await save_to_csv(extracted_memory)
                    return extracted_memory
                else:
                    print("思い出は抽出されませんでした（応答が空でした）。")
                    return None

        except httpx.HTTPStatusError as e:
            print(f"[エラー] 思い出抽出APIエラー: {e.response.status_code}, {e.response.text}")
            return None
        except Exception as e:
            print(f"[エラー] 思い出抽出中に予期せぬエラー: {e}")
            return None

    async def save_to_csv(memories_string: str):

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
        

    # --- メイン処理開始 ---
    await checkLastDate()
    
    try:
        while True:
            # WebSocketからデータを受け取る
            data_bytes = await websocket.receive_bytes() 
            if llm_wating:
                continue 
            
            # VAD処理
            audio_buffer.extend(data_bytes)
            audio_int16 = np.frombuffer(data_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            speech_dict = vad_iterator(torch.from_numpy(audio_float32), return_seconds=True)
            
            if speech_dict and 'end' in speech_dict:
                print("発話終了を検出。文字起こしを実行します...")
                llm_wating = True
                await asyncio.sleep(0.3) # わずかな遅延を挿入

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
        await getNewMemory(test_chat_history)
        
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