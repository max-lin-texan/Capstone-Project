import numpy as np
import pandas as pd

# 1. 設定隨機種子以確保結果可複現，並生成 Shape 為 (20, 6, 10) 的隨機 Tensor
np.random.seed(42)
data_3d = np.random.rand(20, 6, 10).round(3)

# 2. 展平為 2D 表格 (20, 60)
data_2d = data_3d.reshape(20, -1)

# 3. 建立 60 個欄位名稱 (T0_F0 ~ T5_F9)
columns = [f"T{t}_F{f}" for t in range(6) for f in range(10)]

# 4. 建立 DataFrame 並加入 sample_id
df = pd.DataFrame(data_2d, columns=columns)
df.insert(0, "sample_id", range(20))

# 5. 匯出為 CSV 檔
df.to_csv("tensor_20x6x10.csv", index=False)
print("檔案已成功生成並儲存為 'tensor_20x6x10.csv'！")