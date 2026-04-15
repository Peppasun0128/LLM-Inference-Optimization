# LLM 推論核心實作與生成策略深度調優
**Project: LLM Inference Engine Development & Generative Logic Optimization**

## 1. 專案概述 (Executive Summary)
本專案透過在 **Docker** 環境中部署 **Meta Llama-3.2-3B** 模型，完整復現 LLM 推論引擎的底層邏輯。透過量化分析參數變化對生成文本的影響，優化了模型在有限參數規模下的生成表現。

---

## 2. 系統架構與基礎設施 (Infrastructure & Architecture)
* **容器化技術**: 使用 Docker 封裝開發環境，解決 GPU 驅動依賴。
* **開發工具**: VS Code Remote-SSH (Linux Server) 遠端運算。

![Docker 環境安裝與啟動](images/image5.png)

*【圖說】：展示使用 Docker 進行環境隔離與環境安裝，確保推論環境的一致性。*

---

## 3. 核心算法實作：解碼與採樣策略

### 3.1 數學邏輯：從 Logits 到 Probability
手刻 `generate` 迴圈，實作 Softmax 函數並引入溫度參數 $T$，精確控制機率分佈。

![LLM 推論運作流程圖](images/image2.png)

*【圖說】：實作底層推論流程。模型將輸入 Token 轉換為 Logits 後，經過 Softmax 轉換為機率分佈。*

![Softmax 數學公式](images/image8.png)

![Temperature 實驗數據分析](images/image7.png)

*【圖說】：Temperature 實驗數據。$T \to 0$ 時機率集中於最高項（Greedy）；$T$ 升高時分佈平滑化，增加隨機性。*

### 3.2 自適應濾波器 (Sampling Filters)
實作 Top-K、Top-P 及最具適應性的 **Min-P** 演算法。

![採樣過濾器對比結果](images/image1.png)

*【圖說】：對比不同分布下的過濾效果，驗證各採樣策略對 Token 選擇的影響。*

---

## 4. 服務體驗與故障排除 (Service & Troubleshooting)

### 4.1 解決生成跳針：懲罰機制
實作 Repetition / Frequency / Presence Penalty，解決 LLM 無限重複迴圈。

![Penalty 懲罰機制實驗數據](images/image3.png)

*【圖說】：透過 Penalty 壓低重複 Token 權重，引導模型轉換詞彙。*

### 4.2 案例分析：高隨機性下的語言修復
**問題**：高溫設定出現語言漂移。
**對策**：結合 Top-P 採樣策略，成功恢復邏輯軌道。

![語言漂移與修復對比](images/image9.png)

*【圖說】：故障排除實錄。透過加入 Penalty 與 Sampling Filter，成功讓模型在亂碼後恢復正常中文回應。*

---

## 5. 進階策略與人文交匯
身為具備教育背景的開發者，我關注技術對「使用者感受」的影響。在「貓咪 Rap」實驗中，透過三種參數策略驗證了如何精確配置模型參數。

![進階策略設計與要求](images/image4.png)

*【圖說】：設計三種不同風格的生成策略，追求技術穩定與文創表達的平衡。*

![模型最終生成成果](images/image6.png)

*【圖說】：最終生成成果展示，展現模型在調優後的流暢對話能力。*

---

## 6. 結論與展望
本專案展現了從基礎設施到演算法微調的全方位能力。未來計畫整合 **RAG (檢索增強生成)** 技術，進一步優化 AI 服務的資訊正確性。
