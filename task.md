# 本輪研究任務

說明：

- 這份 `task.md` 只放本輪真正要執行的任務，不一次塞入過多想法。
- 每個任務做完都要打勾，並在後方補 `Performance` 或簡短結論。
- 若只是想到但還沒排入本輪，請不要先寫進這裡。
- 若同一主題需要更細的參數展開，先完成上層探索，再補開下層任務。

## 一、標記設計

目標：先確認 `60d / +8% / -4% / drop-neutral` 是否合理，避免後面所有實驗建立在不好的標記上。

- [x] 統計目前 `positive / negative / neutral` 比例。Performance: `positive=2023`, `negative=2346`, `neutral=935`, `total=5304`.
- [x] 檢查 `drop-neutral` 後剩餘樣本數是否足夠。Performance: `drop_neutral_rows=4369`，約保留 `82.4%` 樣本。
- [x] 比較 `drop-neutral` 與 `keep-all binary` 的 class balance。Performance: `drop-neutral` 正類率 `0.4630`；`keep-all binary` 正類率 `0.3814`。
- [x] 比較 `60d +8%/-4%` 與 `60d +10%/-5%` 的標籤分布。Performance: `+10/-5` 的 `neutral_rate=0.3064`，明顯高於目前設定的 `0.1763`。
- [x] 比較 `60d +8%/-4%` 與 `60d +6%/-3%` 的標籤分布。Performance: `+6/-3` 的 `neutral_rate=0.0626`，但正類率降到 `0.4429`。
- [x] 比較 `40d +8%/-4%` 與 `60d +8%/-4%` 的標籤分布。Performance: `40d` 的 `neutral_rate=0.3173`，高於 `60d` 的 `0.1763`。

## 二、基準模型

目標：建立一個穩定、可重跑的 baseline，後面所有改動都和它比較。

- [x] 跑一次目前 baseline，記錄完整指標。Performance: `validation_f1=0.5765`, `validation_bal_acc=0.5102`, `validation_accuracy=0.4150`, `test_f1=0.8129`, `test_bal_acc=0.5526`, `threshold=0.491`.
- [x] 重跑 baseline 第二次，確認結果一致。Performance: 第二次結果完全相同，確認可重跑。
- [x] 固定 threshold `0.50` 比較 baseline。Performance: `validation_f1=0.1433`, `validation_bal_acc=0.4979`，明顯劣於自動 threshold。
- [x] 比較 `threshold_steps=401` 與 `threshold_steps=801`。Performance: 兩者都選到 `threshold=0.491`，`validation_f1=0.5765`，沒有差異。
- [x] 比較 best epoch by `validation_f1` 與 best epoch by `validation_bal_acc`。Performance: 兩者都選到 `epoch=11`，結果一致。

## 三、特徵擴充

目標：先做最有希望的中期特徵，不一次展開太多細碎組合。

- [x] 加入 `ret_60`。Performance: `validation_f1=0.0000`, `validation_bal_acc=0.5000`, `test_f1=0.0000`。
- [x] 加入 `drawdown_60`。Performance: `validation_f1=0.0000`, `validation_bal_acc=0.5000`, `test_f1=0.0000`。
- [x] 加入 `volatility_20`。Performance: `validation_f1=0.5733`, `validation_bal_acc=0.5013`, `test_f1=0.8143`。
- [x] 加入 `volume_vs_60`。Performance: `validation_f1=0.0000`, `validation_bal_acc=0.5000`, `test_f1=0.0000`。
- [x] 加入 `sma_gap_60`。Performance: `validation_f1=0.0000`, `validation_bal_acc=0.5000`, `test_f1=0.0000`。
- [x] 加入 `range_z_20`。Performance: `validation_f1=0.5758`, `validation_bal_acc=0.5064`, `test_f1=0.8173`。
- [x] 加入 `gap_up_flag`。Performance: `validation_f1=0.5752`, `validation_bal_acc=0.5077`, `test_f1=0.8162`。
- [x] 加入 `gap_down_flag`。Performance: `validation_f1=0.5771`, `validation_bal_acc=0.5115`, `test_f1=0.8164`。
- [x] 加入 `inside_bar`。Performance: `validation_f1=0.5771`, `validation_bal_acc=0.5115`, `test_f1=0.8137`。
- [x] 加入 `outside_bar`。Performance: `validation_f1=0.5765`, `validation_bal_acc=0.5102`, `test_f1=0.8159`。

## 四、特徵替換

目標：找出哪些新特徵值得取代舊特徵，而不是只是無限制往上疊。

- [x] 用 `drawdown_60` 替換 `drawdown_20`。Performance: `validation_f1=0.0000`, `validation_bal_acc=0.5000`, `test_f1=0.0000`。
- [x] 用 `volatility_20` 替換 `volatility_10`。Performance: `validation_f1=0.5739`, `validation_bal_acc=0.5026`, `test_f1=0.8166`。
- [x] 用 `sma_gap_60` 替換 `sma_gap_20`。Performance: `validation_f1=0.0000`, `validation_bal_acc=0.5000`, `test_f1=0.0000`。
- [x] 用 `volume_vs_60` 替換 `volume_vs_20`。Performance: `validation_f1=0.0000`, `validation_bal_acc=0.5000`, `test_f1=0.0000`。
- [x] 用 `range_z_20` 替換 `range_pct`。Performance: `validation_f1=0.5758`, `validation_bal_acc=0.5064`, `test_f1=0.8158`。

## 五、互動項

目標：只測少數高價值 interaction，不做無限制暴力組合。

- [x] 測 `gap_up_flag:drawdown_20`。Performance: `validation_f1=0.5752`, `validation_bal_acc=0.5077`, `test_f1=0.8162`。
- [x] 測 `gap_up_flag:volume_vs_20`。Performance: `validation_f1=0.5752`, `validation_bal_acc=0.5077`, `test_f1=0.8151`。
- [x] 測 `drawdown_20:volume_vs_20`。Performance: `validation_f1=0.5853`, `validation_bal_acc=0.5346`, `test_f1=0.8092`。
- [x] 測 `ret_20:drawdown_20`。Performance: `validation_f1=0.5777`, `validation_bal_acc=0.5102`, `test_f1=0.8152`。
- [x] 測 `ret_20:breakout_20`。Performance: `validation_f1=0.5790`, `validation_bal_acc=0.5128`, `test_f1=0.8019`。
- [x] 測 `rsi_14:drawdown_20`。Performance: `validation_f1=0.5791`, `validation_bal_acc=0.5154`, `test_f1=0.7947`。

## 六、參數探索

目標：只做一小圈有根據的參數微調，不做無上限掃描。

- [x] 試 `neg_weight = 1.1`。Performance: `validation_f1=0.5771`, `validation_bal_acc=0.5115`, `test_f1=0.8159`。
- [x] 試 `neg_weight = 1.2`。Performance: `validation_f1=0.5765`, `validation_bal_acc=0.5102`, `test_f1=0.8159`。
- [x] 試 `neg_weight = 1.3`。Performance: `validation_f1=0.5771`, `validation_bal_acc=0.5115`, `test_f1=0.8148`。
- [x] 試 `l2_reg = 5e-4`。Performance: `validation_f1=0.5765`, `validation_bal_acc=0.5102`, `test_f1=0.8129`。
- [x] 試 `l2_reg = 2e-3`。Performance: `validation_f1=0.5765`, `validation_bal_acc=0.5102`, `test_f1=0.8129`。
- [x] 試 `learning_rate = 0.01`。Performance: `validation_f1=0.5771`, `validation_bal_acc=0.5115`, `test_f1=0.8137`。
- [x] 試 `learning_rate = 0.03`。Performance: `validation_f1=0.5765`, `validation_bal_acc=0.5102`, `test_f1=0.8144`。

## 七、驗證穩定性

目標：避免只是在單一 validation 區段看起來漂亮。

- [x] 做一次 3-fold walk-forward validation。Performance: fold `validation_f1=[0.6084, 0.5765, 0.8233]`。
- [x] 做一次 4-fold walk-forward validation。Performance: fold `validation_f1=[0.5043, 0.5214, 0.6326, 0.6743]`。
- [x] 比較不同 fold 的 threshold 穩定性。Performance: 3-fold threshold `[0.471, 0.491, 0.491]`；4-fold threshold `[0.300, 0.496, 0.492, 0.427]`。
- [x] 比較不同 fold 的 `validation_f1` 波動。Performance: 3-fold 範圍約 `0.5765 -> 0.8233`；4-fold 範圍約 `0.5043 -> 0.6743`，穩定性普通。
- [x] 重建資料後再跑 baseline。Performance: refresh 後 baseline 仍為 `validation_f1=0.5765`, `validation_bal_acc=0.5102`, `test_f1=0.8129`。
- [x] 用 3 個 seed 重跑目前最佳設定。Performance: seed `1/2/3` 全部一致，`validation_f1=0.5765`, `test_f1=0.8129`。

## 八、交易解讀

目標：讓模型指標能對應到真正的買入決策，而不是只看分類分數。

- [x] 統計預測為正類時的平均 `future_return_60`。Performance: baseline threshold `0.491` 下，`test_avg_return=8.99%`。
- [x] 統計預測為正類時的勝率。Performance: baseline threshold `0.491` 下，預測正類的 `win_rate=0.8462`。
- [x] 比較不同 threshold 下的正類比例。Performance: `0.45 -> 1.0000`, `0.49 -> 0.9557`, `0.50 -> 0.0474`, `0.55 -> 0.0000`。
- [x] 比較不同 threshold 下的平均報酬。Performance: `0.45 -> 8.27%`, `0.49 -> 8.85%`, `0.50 -> 9.97%`, `0.55 -> 0%`。
- [x] 寫出一版簡單交易解讀摘要。Performance: 目前 baseline 對正類預測過多，導致 `validation_f1` 偏弱；雖然預測正類的平均 60 日報酬高，但暫時不能當成穩定交易訊號。

## 下一輪候選方向

這裡暫時不列具體任務，只保留方向提示：

- barrier 設計第二輪細化
- 中期回撤類特徵深化
- ranking / 回歸版本
- 簡易回測框架

---

# 下一輪研究任務

## 一、標記設計第二輪

- [x] 正式比較 `drop-neutral` 與 `keep-all binary` 的 baseline 表現。Performance: `keep-all binary` 為 `validation_f1=0.5341`, `validation_bal_acc=0.5052`, `test_f1=0.7001`, `test_bal_acc=0.4993`，全面弱於目前 `drop-neutral` baseline。
- [x] 正式比較 `60d +8%/-4%` 與 `60d +6%/-3%` 的 baseline 表現。Performance: `60d +6%/-3%` 為 `validation_f1=0.6080`, `validation_bal_acc=0.5319`, `test_f1=0.7231`, `test_bal_acc=0.4802`，validation 提升但 test 明顯退化。
- [x] 正式比較 `60d +8%/-4%` 與 `60d +10%/-5%` 的 baseline 表現。Performance: `60d +10%/-5%` 為 `validation_f1=0.5167`, `validation_bal_acc=0.5378`, `test_f1=0.7277`, `test_bal_acc=0.5602`，validation_f1 明顯低於目前預設。

## 二、目前最佳模型深化

- [ ] 在目前最佳設定上測 `neg_weight = 1.1`。Performance:
- [ ] 在目前最佳設定上測 `neg_weight = 1.2`。Performance:
- [ ] 在目前最佳設定上測 `learning_rate = 0.01`。Performance:
- [ ] 在目前最佳設定上比較 `threshold_steps=401` 與 `801`。Performance:

## 三、最佳方向延伸

- [ ] 在目前最佳設定上加入 `gap_down_flag`。Performance:
- [ ] 在目前最佳設定上加入 `inside_bar`。Performance:
- [ ] 在目前最佳設定上加入 `range_z_20`。Performance:
- [ ] 在目前最佳設定上加入 `gap_down_flag:drawdown_20`。Performance:
- [ ] 在目前最佳設定上加入 `range_z_20:drawdown_20`。Performance:

## 四、交易規則校準

- [ ] 比較 `threshold=0.42`、`0.45`、`0.49` 的交易解讀。Performance:
- [ ] 只交易最高信心 `top 20%` 樣本並統計表現。Performance:
- [ ] 只交易最高信心 `top 10%` 樣本並統計表現。Performance:

## 五、問題排查

- [ ] 釐清為何 `ret_60`、`drawdown_60`、`sma_gap_60`、`volume_vs_60` 加入後會退化到幾乎全負類。Performance:
- [ ] 檢查 validation 與 test 落差過大的原因，特別是正類比例與 threshold 行為。Performance:
