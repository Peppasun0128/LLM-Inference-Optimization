# LLM 推論核心實作與生成策略深度調優
**Project: LLM Inference Engine Development & Generative Logic Optimization**

## 1. 專案概述 (Executive Summary)
本專案透過在 **Docker** 容器化環境中部署 **Meta Llama-3.2-3B** 模型，完整復現 LLM 推論引擎的底層邏輯。專案核心在於手刻實作 **Softmax 溫度控制**、**多樣化採樣濾波器 (Top-K/P, Min-P)** 及 **懲罰機制 (Penalty)**。透過量化分析參數變化對生成文本（如創意寫作、邏輯推論）的影響，優化了模型在有限參數規模下的生成表現。

---

## 2. 系統架構與基礎設施 (Infrastructure & Architecture)
本專案採用業界標準的 DevOps 工作流，確保環境的可移植性與運算效能。

* **容器化技術**: 使用 Docker 封裝開發環境，解決 CUDA 版本與 GPU 驅動的複雜依賴。
* **效能優化**: 針對 Llama 3.2 結構，手動處理 **BFloat16** 運算精度。
* **開發工具**: VS Code Remote-SSH (Linux Server) 進行遠端運算。

> **📸 展示 Docker 啟動指令：image_7.png**
> **【圖說】**：展示使用 Docker 進行環境隔離與硬體掛載（GPU Volumes），確保模型推論環境的一致性與可移植性。

---

## 3. 核心算法實作：解碼與採樣策略

### 3.1 數學邏輯：從 Logits 到 Probability
不依賴封裝好的高階函式，手刻 `generate` 迴圈。透過實作 **Softmax** 函數並引入溫度參數 $T$，精確控制機率分佈的平滑度。

$$P_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

> **📸 展示 Softmax 公式：image_10.png**
> **📸 展示 Temperature 實驗數據表與分析：image_9.png**
> **📸 展示推論運作流程圖：image_4.png**
> **【圖說】**：實作底層推論流程。數據顯示溫度 $T$ 越高，創造力越高；$T \to 0$ 時則轉向決定性的 Greedy Search。

### 3.2 自適應採樣策略 (Advanced Sampling)
為了在生成的多樣性與邏輯性間取得平衡，實作了以下過濾器：
* **Top-K / Top-P (Nucleus Sampling)**: 限制候選詞範圍。
* **Min-P (Adaptive Cutoff)**: 根據最高機率動態調整門檻（最具適應性的現代解法）。

> **📸 展示 LAB-3 採樣對比結果：image_3.png**
> **【圖說】**：對比不同分布下的過濾效果。實驗證明 Min-P 較 Top-P 能更有效地在維持語意流暢的同時，過濾低機率噪訊。

---

## 4. 服務體驗與故障排除 (Service & Troubleshooting)

### 4.1 解決生成跳針：懲罰機制 (Penalties)
針對 LLM 常見的「跳針（無限重複）」問題，實作 **Repetition、Frequency 與 Presence Penalty**。

> **📸 展示 LAB-5 懲罰機制數據表：image_5.png**
> **【圖說】**：懲罰機制實測。透過Penalty 壓低重複 Token 權重，引導模型轉換詞彙。

### 4.2 案例分析：高隨機性下的語言修復
**問題再現**：在高溫設定 (T=1.2) 下，模型出現語言漂移（Language Drift），輸出非預期語言或亂碼。

> **📸 展示故障排除前的亂碼圖：image_11.png**
> **【圖說】**：故障排除前。高溫 T=1.2 設定導致 Llama-3.2-3B 模型邏輯崩潰，噴出泰文、德文、程式碼片段與無意義字符。

**優化對策**：引入 **Nucleus Sampling (Top-P=0.5)** 嚴格截斷長尾分佈的雜訊詞彙。

> **📸 展示故障排除後的修復圖：image_8.png**
> **【圖說】**：故障排除後。套用優化後的 Top-P=0.9 採樣策略（與 輕微 Top-K 安全網），成功截斷低機率噪訊，將模型輸出引導回邏輯清晰、語意通順的繁體中文語境。

---

## 5. 跨域應用：技術與人文的交匯 (Service Innovation)
身為具備語言教育背景的開發者，我特別關注技術對「使用者感受」的影響。在「貓咪 Rap 寫作」實驗中，我測試了三種參數組合：策略 1 (創意感)、策略 2 (穩定感)、策略 3 (平衡感)。

> **📸 展示 LAB-Summary 進階策略分析圖：image_1.png**
> **【圖說】**：進階策略設計與驗證。展示如何根據不同的服務需求（如：創意行銷 vs. 精準客服），精確配置模型參數以達到最佳使用者感受。

---

## 6. 結論與展望 (Conclusion)
本專案展現了從基礎設施部署到演算法微調的全方位能力。未來計畫整合 **RAG (檢索增強生成)** 架構，進一步優化 AI 服務的資訊正確性。
