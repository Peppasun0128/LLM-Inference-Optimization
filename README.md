# LLM 推論核心實作與生成策略深度調優
**Project: LLM Inference Engine Development & Generative Logic Optimization**

## 1. 專案概述 (Executive Summary)
本專案透過在 **Docker** 容器化環境中部署 **Llama-3.2-3B** 模型，完整復現 LLM 推論引擎的底層邏輯。專案核心在於手刻實作 **Softmax 溫度控制**、**多樣化採樣濾波器 (Top-K/P, Min-P)** 及 **懲罰機制 (Penalty)**。透過量化分析參數變化對生成文本的影響，優化了模型在有限參數規模下的生成表現。

---

## 2. 系統架構與基礎設施 (Infrastructure & Architecture)
* **容器化技術**: 使用 Docker 封裝開發環境，解決 CUDA 與 GPU 驅動依賴。
* **效能優化**: 針對 Llama 3.2 結構，手動處理 **BFloat16** 運算精度。
* **開發工具**: VS Code Remote-SSH (Linux Server) 遠端運算。

> **📸 此處建議放置：image_d24081.png (Docker 啟動截圖)**
> **【圖說】**：展示使用 Docker 進行環境隔離與硬體掛載，確保推論環境的一致性。

---

## 3. 核心算法實作：解碼與採樣策略
### 3.1 數學邏輯：從 Logits 到 Probability
手刻 `generate` 迴圈，實作 Softmax 函數並引入溫度參數 $T$，精確控制機率分佈。

> **📸 此處建議放置：image_d1bcbe.png (推論流程圖)**
> **【圖說】**：實作底層推論流程。數據顯示溫度 $T$ 越高，創造力越高；$T \to 0$ 時則轉向穩定。

### 3.2 自適應濾波器 (Sampling Filters)
實作 Top-K、Top-P (Nucleus Sampling) 及最具適應性的 **Min-P** 演算法。

> **📸 此處建議放置：image_de7f00.png (LAB-3 採樣對比結果)**
> **【圖說】**：對比不同分布下的過濾效果，Min-P 在語意流暢度與噪訊過濾間取得最佳平衡。

---

## 4. 服務體驗與故障排除 (Service & Troubleshooting)
### 4.1 解決生成跳針：懲罰機制
實作 Repetition / Frequency / Presence Penalty，解決 LLM 無限重複迴圈的痛點。

> **📸 此處建議放置：image_e0457a.png (LAB-5 數據)**
> **【圖說】**：透過 Penalty 壓低重複 Token 權重，引導模型轉換詞彙。

### 4.2 案例分析：高隨機性下的語言修復
**問題**：高溫設定 (T=1.2) 出現語言漂移。
**對策**：結合 Min-P (0.05) 與 Top-P (0.9) 雙重機制，成功恢復邏輯軌道。

> **📸 此處建議放置：image_de1d68.png (亂碼) 與 image_de1d8a.png (修復) 並列圖**
> **【圖說】**：故障排除實錄，展示模型從崩潰到修復的完整調優過程。

---

## 5. 跨域應用：技術與人文的交匯 (Service Innovation)
身為具備教育背景的開發者，我特別關注技術對「使用者感受」的影響。在「貓咪 Rap」實驗中，透過三種參數策略（創意流、嚴謹流、平衡流）的對比，驗證了如何根據服務需求精確配置模型參數。

---

## 6. 結論與展望
本專案展現了從基礎設施到演算法微調的全方位能力。未來計畫整合 **RAG (檢索增強生成)** 技術，進一步優化 AI 服務的資訊正確性。
