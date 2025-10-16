GPUリアルタイム文字起こし＆チャットアプリケーション
このプロジェクトは、マイクからの音声をリアルタイムで文字起こしし、そのテキストをDifyのチャットAPIに送信するFastAPIアプリケーションをDockerで実行するためのものです。NVIDIA GPUを活用して、高速かつ高精度な音声認識を実現します。

特徴
リアルタイム音声認識: faster-whisperとSilero VADを使用して、発話の終了を検出し、リアルタイムで文字起こしを行います。

GPUアクセラレーション: NVIDIA GPUを利用して、large-v3モデルでも高速な処理が可能です。

会話の継続: conversation_idをファイルに保存し、アプリケーションを再起動しても会話を続けることができます。

簡単な環境構築: DockerとDocker Composeを使用するため、依存関係のインストールや環境設定が簡単です。

柔軟な設定: DifyのAPIキーは.envファイルで管理するため、コードを変更する必要がありません。

前提条件
実行するホストマシンに以下のソフトウェアがインストールされている必要があります。

Docker: Docker公式サイト の手順に従ってインストールしてください。

Docker Compose: Docker Desktopには通常含まれています。個別にインストールする場合は公式ドキュメントを参照してください。

NVIDIA GPUドライバ: ご利用のGPUに対応した最新のドライバをインストールしてください。

NVIDIA Container Toolkit: DockerコンテナからGPUを利用するために必須です。インストールガイドに従って設定してください。

実行方法
1. リポジトリのクローン（またはファイルのダウンロード）
まず、すべてのファイル (main.py, Dockerfile, docker-compose.yml, requirements.txt, .env.example, README.md) を同じディレクトリに配置します。

2. 環境変数の設定
.env.example ファイルをコピーして .env という名前のファイルを作成します。

cp .env.example .env

次に、作成した .env ファイルをお好みのテキストエディタで開き、YOUR_DIFY_API_KEY_HERE の部分を実際のDify APIキーに置き換えてください。

DIFY_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

3. Dockerイメージのビルドとコンテナの起動
ターミナルでプロジェクトのディレクトリに移動し、以下のコマンドを実行します。

docker-compose up --build

初回はベースイメージのダウンロードとライブラリのインストールに時間がかかります。ビルドが完了すると、コンテナが起動し、FastAPIサーバーがポート 8000 でリッスンを開始します。

-d オプションを付けて docker-compose up --build -d とすると、バックグラウンドでコンテナが起動します。

4. 動作確認
サーバーが起動したら、WebSocketクライアント（Pythonスクリプトやウェブサイトなど）から ws://localhost:8000/ws/transcribe に接続し、マイクからの音声データ（16kHz, 16-bit PCM）を送信してください。ターミナルに文字起こしの結果やDifyからの応答が表示されます。

ログの確認
コンテナがバックグラウンドで動作している場合、以下のコマンドでログを確認できます。

docker-compose logs -f

-f オプションを付けると、ログをリアルタイムで追跡できます。

コンテナの停止
アプリケーションを停止するには、ターミナルで Ctrl + C を押すか、以下のコマンドを実行します。

docker-compose down

このコマンドはコンテナを停止し、削除します。ただし、docker-compose.ymlで設定したボリューム（./dataディレクトリ）はホストマシン上に残るため、会話データは失われません。