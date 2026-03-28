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

- [x] 在目前最佳設定上測 `neg_weight = 1.1`。Performance: `validation_f1=0.5851`, `validation_bal_acc=0.5320`, `test_f1=0.8115`, `test_bal_acc=0.5623`，test 略好但 validation_f1 仍低於目前最佳。
- [x] 在目前最佳設定上測 `neg_weight = 1.2`。Performance: `validation_f1=0.5853`, `validation_bal_acc=0.5346`, `test_f1=0.8081`, `test_bal_acc=0.5637`，validation 持平但 test_f1 較弱。
- [x] 在目前最佳設定上測 `learning_rate = 0.01`。Performance: `validation_f1=0.5845`, `validation_bal_acc=0.5307`, `test_f1=0.8074`, `test_bal_acc=0.5613`，未優於目前最佳。
- [x] 在目前最佳設定上比較 `threshold_steps=401` 與 `801`。Performance: `threshold=0.422`、`validation_f1=0.5853`、`test_f1=0.8092` 完全一致，沒有額外收益。

## 三、最佳方向延伸

- [x] 在目前最佳設定上加入 `gap_down_flag`。Performance: `validation_f1=0.5836`, `validation_bal_acc=0.5327`, `test_f1=0.8058`, `test_bal_acc=0.5614`，未優於目前最佳。
- [x] 在目前最佳設定上加入 `inside_bar`。Performance: `validation_f1=0.5849`, `validation_bal_acc=0.5327`, `test_f1=0.8081`, `test_bal_acc=0.5589`，未優於目前最佳。
- [x] 在目前最佳設定上加入 `range_z_20`。Performance: `validation_f1=0.5840`, `validation_bal_acc=0.5262`, `test_f1=0.8092`, `test_bal_acc=0.5552`，test_f1 持平但 validation 與平衡度較弱。
- [x] 在目前最佳設定上加入 `gap_down_flag:drawdown_20`。Performance: `validation_f1=0.5836`, `validation_bal_acc=0.5327`, `test_f1=0.8070`, `test_bal_acc=0.5626`，沒有形成有效突破。
- [x] 在目前最佳設定上加入 `range_z_20:drawdown_20`。Performance: `validation_f1=0.5859`, `validation_bal_acc=0.5359`, `test_f1=0.8054`, `test_bal_acc=0.5579`，validation 小幅創高但 test 退化，不升級為正式最佳。

## 四、交易規則校準

- [x] 比較 `threshold=0.42`、`0.45`、`0.49` 的交易解讀。Performance: `0.42` 最接近目前最佳，`test_f1=0.8085`, `avg_return=8.83%`；`0.45` 雖把單筆平均報酬拉到 `9.28%`，但 `test_f1` 降到 `0.6279`；`0.49` 僅交易 `1.38%` 樣本，`test_f1=0.0310`，過度保守。
- [x] 只交易最高信心 `top 20%` 樣本並統計表現。Performance: 約對應門檻 `0.4661`，`selected_count=131`, `avg_return=9.09%`, `win_rate=0.6870`，但 `test_f1=0.3141`, `test_bal_acc=0.5051`，不適合直接取代現行規則。
- [x] 只交易最高信心 `top 10%` 樣本並統計表現。Performance: 約對應門檻 `0.4735`，`selected_count=66`, `avg_return=7.85%`, `win_rate=0.6970`，`test_f1=0.1811`，過度稀疏且報酬也未更好。

## 五、問題排查

- [x] 釐清為何 `ret_60`、`drawdown_60`、`sma_gap_60`、`volume_vs_60` 加入後會退化到幾乎全負類。Performance: 根因不是特徵本身，而是這四個 `60` 日特徵在 train split 各殘留 `8` 個 NaN；先前標準化時 NaN 直接污染整欄，導致 logits 與 threshold 搜尋失真，表面上變成幾乎全負類。已在 `train.py` 加入 NaN guard，未來會直接報錯而不是產生假結果。
- [x] 檢查 validation 與 test 落差過大的原因，特別是正類比例與 threshold 行為。Performance: 目前最佳設定在 validation 的實際正類率僅 `0.4012`，test 則升到 `0.6758`；固定 threshold `0.422` 下，模型在兩段都預測約九成正類，但 validation precision 只有 `0.4191`，test precision 升到 `0.7047`。主因較像時段分布轉移與 test 區間更偏多頭，而不是模型在 test 真正更穩定。

---

# 再下一輪研究任務

## 一、長週期特徵修正重跑

- [x] 補上實驗特徵的 NaN 處理策略，正式重跑 `ret_60`。Performance: NaN-clean 資料集為 `4347` 列，`ret_60` 重新變成有效特徵，`validation_f1=0.5868`, `validation_bal_acc=0.5345`, `test_f1=0.8128`, `test_bal_acc=0.5718`，整體優於 NaN-clean baseline。
- [x] 補上實驗特徵的 NaN 處理策略，正式重跑 `drawdown_60`。Performance: `validation_f1=0.5824`, `validation_bal_acc=0.5333`, `test_f1=0.8078`, `test_bal_acc=0.5636`，未優於 NaN-clean baseline。
- [x] 補上實驗特徵的 NaN 處理策略，正式重跑 `sma_gap_60` 與 `volume_vs_60`。Performance: `sma_gap_60` 為 `validation_f1=0.5917`, `validation_bal_acc=0.5428`, `test_f1=0.8083`, `test_bal_acc=0.5768`，validation 與平衡度最強；`volume_vs_60` 為 `validation_f1=0.5811`, `validation_bal_acc=0.5224`, `test_f1=0.8084`, `test_bal_acc=0.5516`，仍偏弱。

## 二、目前最佳設定再驗證

- [x] 對 `neg_weight = 1.1` 做 seed 與 walk-forward 驗證，確認 test 改善是否可重現。Performance: seed `1/2/3` 完全一致，但 forward folds 的 `test_f1=[0.6065, 0.7468, 0.0000]`、`test_bal_acc=[0.4986, 0.5262, 0.0000]` 波動極大，test 小幅改善不具穩定性。
- [x] 對 `range_z_20:drawdown_20` 做 seed 與 walk-forward 驗證，確認 validation 小幅提升是否可信。Performance: seed `1/2/3` 完全一致，但 forward folds 的 `test_f1=[0.6047, 0.7526, 0.0000]`、`test_bal_acc=[0.5002, 0.5381, 0.0000]` 同樣不穩定，validation 小升幅不足以升級。

## 三、交易框架下一步

- [x] 為目前最佳設定建立簡單回測摘要，至少統計命中率、平均報酬與最大回撤。Performance: 以 NaN-clean baseline 的 `threshold=0.433` 做簡化逐筆回測摘要，`selected_count=588`, `hit_rate=0.7075`, `avg_return=9.02%`, `max_drawdown=-84.70%`；顯示目前規則雖命中率高，但訊號過密時資金曲線品質很差。
- [x] 比較「現行 threshold 規則」與「top 20% ranking 規則」在簡單回測下的差異。Performance: `top 20%` 規則選出 `131` 筆、`hit_rate=0.6870`, `avg_return=9.19%`, `max_drawdown=-35.33%`；雖命中率略低，但回撤明顯比現行 threshold 規則低。

---

# 再下下一輪研究任務

## 一、長週期特徵正式升級驗證

- [x] 以 `ret_60` 為新候選最佳，做 seed 與 walk-forward 驗證。Performance: seed `1/2/3` 完全一致，當前切分下 `validation_f1=0.5868`, `test_f1=0.8128`, `test_bal_acc=0.5718`；forward folds 前兩折 `test_f1=0.5748 -> 0.8140`，顯示可用但仍受時段影響。
- [x] 以 `sma_gap_60` 為新候選最佳，做 seed 與 walk-forward 驗證。Performance: seed `1/2/3` 完全一致，當前切分下 `validation_f1=0.5917`, `validation_bal_acc=0.5428`, `test_bal_acc=0.5768` 最強；forward folds 前兩折 `test_f1=0.5242 -> 0.8107`，穩定性仍受 regime 影響。
- [x] 測試 `ret_60 + drawdown_20:volume_vs_20` 是否可疊加。Performance: 指標與 `ret_60` 完全一致，因為目前模型預設就已含 `drawdown_20:volume_vs_20` interaction，沒有新增效果。
- [x] 測試 `sma_gap_60 + drawdown_20:volume_vs_20` 是否可疊加。Performance: 指標與 `sma_gap_60` 完全一致，原因同上，屬於重複設定而非新突破。

## 二、交易規則深化

- [x] 建立非重疊持倉的簡單回測，重新估算目前最佳規則的最大回撤。Performance: 改成非重疊持倉後，NaN-clean baseline `threshold rule` 只產生 `11` 筆交易，`hit_rate=0.7273`, `avg_return=8.01%`, `max_drawdown=-5.41%`；先前的 `-84.70%` 主要是重疊持倉假設造成的失真。
- [x] 比較 `ret_60` 模型在 `threshold rule` 與 `top 20% ranking rule` 下的回測摘要。Performance: `threshold rule` 為 `11` 筆、`hit_rate=0.7273`, `avg_return=8.01%`, `max_drawdown=-5.41%`；`top 20%` 為 `9` 筆、`hit_rate=0.7778`, `avg_return=9.91%`, `max_drawdown=-6.80%`，報酬與命中率較高，但回撤略大且樣本更少。

## 三、時段轉移排查

- [x] 比較 validation 與 test 區間的實際 barrier 命中分布，確認是否存在 regime shift。Performance: validation 正類率僅 `0.4003`、平均 60 日報酬 `1.41%`，test 正類率升到 `0.6753`、平均 60 日報酬 `8.27%`，明顯存在多頭 regime shift。
- [x] 測試改用較晚起始年份訓練，是否能縮小 validation/test 落差。Performance: `2012` 與 `2016` 起訓雖讓 `f1` 看起來升高，但 `validation_bal_acc`/`test_bal_acc` 都掉到 `0.5000`，本質上只是更偏向全正類，沒有真的縮小落差。

---

# 再下下下一輪研究任務

## 一、候選最佳正面對決

- [ ] 正式比較 `ret_60` 與 `sma_gap_60`，加入同一張對照表與交易摘要。Performance:
- [ ] 測試 `ret_60 + sma_gap_60` 是否能同時保留 test_f1 與 validation_bal_acc。Performance:

## 二、交易規則細化

- [ ] 對 `ret_60` 模型測 `top 10%`、`top 15%`、`top 20%` 的非重疊持倉回測。Performance:
- [ ] 對 `sma_gap_60` 模型測 `threshold rule` 與 `top 20% rule` 的非重疊持倉回測。Performance:

## 三、Regime 感知驗證

- [ ] 將 validation/test 按年份切段，統計每段的 barrier 正類率與平均 60 日報酬。Performance:
- [ ] 測試加入簡單 regime 特徵後，是否能降低 validation/test 分布落差。Performance:
