import csv
from datetime import datetime

memory_file_name = "./data/Memory.csv"
fitbit_file_name = "FitBit.csv"

class Memory:
    def __init__(self, title, date, ctent):
        self.title = title
        self.date = dateon
        self.content = content

def insertData(memories):
    """
    Memoryクラスのオブジェクトのリストを受け取り、CSVに追記する
    """
    with open(memory_file_name, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for memory in memories:
            writer.writerow([memory.title, memory.date, memory.content])

# 使用例
mem1 = Memory("初めての旅行", datetime.now().strftime("%Y-%m-%d"), "京都に行ってお寺を巡った")
mem2 = Memory("誕生日", datetime.now().strftime("%Y-%m-%d"), "友達とケーキを食べた")

insertData([mem1, mem2])


if __name__ == "__main__":
    # サンプルデータ作成
    memories = [
        Memory("誕生日", "2025-09-21", "友達とケーキを食べた"),
        Memory("旅行", "2025-09-20", "箱根に温泉旅行に行った")
    ]

    # CSVに書き込む
    insertData(memories)