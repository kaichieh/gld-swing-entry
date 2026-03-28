# 本輪研究任務

說明：

- 這份 `task.md` 只放本輪真正要執行的任務，不一次塞入過多想法。
- 每個任務做完都要打勾，並在後方補 `Performance` 或簡短結論。
- 若只是想到但還沒排入本輪，請不要先寫進這裡。
- 若同一主題需要更細的參數展開，先完成上層探索，再補開下層任務。

## 一、標記設計

目標：先確認 `60d / +8% / -4% / drop-neutral` 是否合理，避免後面所有實驗建立在不好的標記上。

- [ ] 統計目前 `positive / negative / neutral` 比例。Performance:
- [ ] 檢查 `drop-neutral` 後剩餘樣本數是否足夠。Performance:
- [ ] 比較 `drop-neutral` 與 `keep-all binary` 的 class balance。Performance:
- [ ] 比較 `60d +8%/-4%` 與 `60d +10%/-5%` 的標籤分布。Performance:
- [ ] 比較 `60d +8%/-4%` 與 `60d +6%/-3%` 的標籤分布。Performance:
- [ ] 比較 `40d +8%/-4%` 與 `60d +8%/-4%` 的標籤分布。Performance:

## 二、基準模型

目標：建立一個穩定、可重跑的 baseline，後面所有改動都和它比較。

- [ ] 跑一次目前 baseline，記錄完整指標。Performance:
- [ ] 重跑 baseline 第二次，確認結果一致。Performance:
- [ ] 固定 threshold `0.50` 比較 baseline。Performance:
- [ ] 比較 `threshold_steps=401` 與 `threshold_steps=801`。Performance:
- [ ] 比較 best epoch by `validation_f1` 與 best epoch by `validation_bal_acc`。Performance:

## 三、特徵擴充

目標：先做最有希望的中期特徵，不一次展開太多細碎組合。

- [ ] 加入 `ret_60`。Performance:
- [ ] 加入 `drawdown_60`。Performance:
- [ ] 加入 `volatility_20`。Performance:
- [ ] 加入 `volume_vs_60`。Performance:
- [ ] 加入 `sma_gap_60`。Performance:
- [ ] 加入 `range_z_20`。Performance:
- [ ] 加入 `gap_up_flag`。Performance:
- [ ] 加入 `gap_down_flag`。Performance:
- [ ] 加入 `inside_bar`。Performance:
- [ ] 加入 `outside_bar`。Performance:

## 四、特徵替換

目標：找出哪些新特徵值得取代舊特徵，而不是只是無限制往上疊。

- [ ] 用 `drawdown_60` 替換 `drawdown_20`。Performance:
- [ ] 用 `volatility_20` 替換 `volatility_10`。Performance:
- [ ] 用 `sma_gap_60` 替換 `sma_gap_20`。Performance:
- [ ] 用 `volume_vs_60` 替換 `volume_vs_20`。Performance:
- [ ] 用 `range_z_20` 替換 `range_pct`。Performance:

## 五、互動項

目標：只測少數高價值 interaction，不做無限制暴力組合。

- [ ] 測 `gap_up_flag:drawdown_20`。Performance:
- [ ] 測 `gap_up_flag:volume_vs_20`。Performance:
- [ ] 測 `drawdown_20:volume_vs_20`。Performance:
- [ ] 測 `ret_20:drawdown_20`。Performance:
- [ ] 測 `ret_20:breakout_20`。Performance:
- [ ] 測 `rsi_14:drawdown_20`。Performance:

## 六、參數探索

目標：只做一小圈有根據的參數微調，不做無上限掃描。

- [ ] 試 `neg_weight = 1.1`。Performance:
- [ ] 試 `neg_weight = 1.2`。Performance:
- [ ] 試 `neg_weight = 1.3`。Performance:
- [ ] 試 `l2_reg = 5e-4`。Performance:
- [ ] 試 `l2_reg = 2e-3`。Performance:
- [ ] 試 `learning_rate = 0.01`。Performance:
- [ ] 試 `learning_rate = 0.03`。Performance:

## 七、驗證穩定性

目標：避免只是在單一 validation 區段看起來漂亮。

- [ ] 做一次 3-fold walk-forward validation。Performance:
- [ ] 做一次 4-fold walk-forward validation。Performance:
- [ ] 比較不同 fold 的 threshold 穩定性。Performance:
- [ ] 比較不同 fold 的 `validation_f1` 波動。Performance:
- [ ] 重建資料後再跑 baseline。Performance:
- [ ] 用 3 個 seed 重跑目前最佳設定。Performance:

## 八、交易解讀

目標：讓模型指標能對應到真正的買入決策，而不是只看分類分數。

- [ ] 統計預測為正類時的平均 `future_return_60`。Performance:
- [ ] 統計預測為正類時的勝率。Performance:
- [ ] 比較不同 threshold 下的正類比例。Performance:
- [ ] 比較不同 threshold 下的平均報酬。Performance:
- [ ] 寫出一版簡單交易解讀摘要。Performance:

## 下一輪候選方向

這裡暫時不列具體任務，只保留方向提示：

- barrier 設計第二輪細化
- 中期回撤類特徵深化
- ranking / 回歸版本
- 簡易回測框架
