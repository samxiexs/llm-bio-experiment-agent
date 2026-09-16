# llm-bio-experiment-agent

**用强化学习教 LLM 做闭环科学实验 —— 项目现状、结果与建议**

本 README 基于 [`result-tex/scientific_experimentation_rl_full_report.tex`](result-tex/scientific_experimentation_rl_full_report.tex)（2026-09 版，15 页）整理，目的是让没有跟完整个过程的人在 20 分钟内搞清楚：**做了什么、结果如何、下一步该干什么**。第 1–4 节忠实转述报告；第 5、6 节是整理者的解读和建议，与报告原文结论分开。

---

## 1. 一句话

> **能不能用 RL 教一个 LLM 学会"做实验"？**

目前的答案：**"告诉模型科学任务要什么"这一半已经解决（自然语言目的 → 可执行 reward，held-out 泛化很强）；"让模型真正根据实验证据来选实验、下结论"这一半还没有做到。** 现有 RL 模型在 benchmark 上的提升主要来自数值捷径（输出先验均值 / 常数），不是利用证据。

## 2. 研究目标

要学的是一个闭环策略

$$
\pi_\theta(e_t \mid q, h_t), \qquad h_t = (D_0, e_1, y_1, \ldots, e_{t-1}, y_{t-1})
$$

- $q$：自然语言描述的科学目的（"估计这个酶的 $K_m$"、"在约束 X 下最大化产量"……）
- $e_t$：第 $t$ 步选择的实验；$y_t$：模拟器返回的结果
- 模型自己决定何时 `STOP`，然后给出最终决策 $d_T$

要求满足四条性质：**目标条件化**（行为随 $q$ 变）、**证据利用**（行为随 $y$ 变）、**适应性**（不同结果 → 不同后续实验）、**效率**（权衡实验成本，知道何时停）。部署时模型是 frozen 的：不允许再有 EIG oracle、DUG oracle、外部 reward model 等外挂。

环境是生化 / 酶动力学类的模拟世界，覆盖六个目标族：F1 机制识别、F2 参数估计（如 $K_m$）、F3 预测、F4 无约束优化、F5 约束优化、F6 假设检验。

## 3. 做了什么

整个项目不是一条路线走到底，而是 **"假设 → 验证/否定 → 转向"** 走了七轮。按时间顺序：

### 3.1 Stage 0 — 因果识别 pilot

**问题**：在最小的因果图识别环境里，epistemic RL（奖励 = 后验熵下降 $r_t^{IG} = H(b_t) - H(b_{t+1})$）能否改变 LLM 的实验选择？

**结论**（$n=200$，BH-FDR 多重校正）：

- 直接 RL：Epistemic-RL 对 Frozen、对 Outcome-RL 都**无显著优势**；表示扰动下的"增益"校正后消失（大量检验中的假阳性）
- 一个约 $1.8\times10^4$ 参数的结构化 MLP **打败所有 LLM 配置**
- 唯一的正信号：**先 SFT 再 RL** 有明显提升，且 SFT + epistemic > SFT + outcome

### 3.2 Stage 1 — 机制消融

**问题**：直接 RL 效果差，是因为动作接口、状态表示、时间信用分配，还是 token 级信用稀释？

方法：Trajectory-Epi / Step-Epi / Step-Epi + action-mask / Epi-Return（长程 $\gamma$ 折扣）四种奖励形态 × Instruct 或 SFT 初始化。表示/接口 `nl / structured` 由预注册规则在独立验证集上选定。

**主要结果**（ID 归一化 regret，越低越好；参照：BED oracle 0.033，结构化 MLP 0.219，Frozen 0.491，Random 0.497）：

| 方法                           | ID regret / argmax match |
| ------------------------------ | ------------------------ |
| Instruct + Outcome             | .492 / .346              |
| Instruct + Trajectory-Epi      | .462 / .376              |
| SFT + Outcome                  | .404 / .431              |
| **SFT + Trajectory-Epi** | **.328 / .510**    |
| **SFT + Epi-Return**     | **.324 / .515**    |

预注册假设：

- H1 epistemic > outcome：支持（效应 $-0.021$，CI $[-0.038, -0.005]$，$p=0.0057$）
- H2 step-level credit 优于 trajectory：**不支持**（$+0.013$，方向反了，$p=0.955$）
- H3 action-token mask 有帮助：**不支持**（效应 $-0.0002$，等于零）
- H4 SFT + Epi > SFT + Outcome：支持（$-0.080$，CI $[-0.093, -0.067]$，$p<10^{-4}$）

**结论**：细粒度时间 / token 信用**不是**瓶颈；SFT 初始化才是最大因素（epistemic 下 $\Delta$regret $-0.136$，outcome 下 $-0.090$）。去掉自由 CoT 反而略好（$-0.034$）；hybrid / structured 状态表示只有微小帮助（$-0.012$ / $-0.008$）。

### 3.3 Round 2 — SFT 初始化研究

**问题**：epistemic RL 的增益 $\Delta_{epi}(C) = J_{EpiRL}(C) - J_{OutcomeRL}(C)$ 如何依赖 SFT checkpoint $C$？

**结论**：

- **倒 U 形**，不是阈值也不是单调 scaling：ckpt500 峰值 $+0.116$（四个 split 一致），ckpt2000 掉约 30%
- 不同 SFT 课程差异巨大：Format-SFT $+0.015 \sim 0.038$，Reasoning-SFT $+0.003 \sim 0.019$，**Oracle-action SFT 峰值 $+0.116$**
- 否定了"能力 = 理解后验 / 假设 / 科学状态"的解释；**见过专家实验动作的偏好**才是关键因素

### 3.4 概念重置

意识到因果 pilot 太窄：不能把"好实验"硬编码成 $\arg\max_e IG(e)$，因为实验的价值取决于科学目的（识别、估参、预测、优化、约束优化、假设检验各不相同）。于是显式引入自然语言目的 $q$，并把问题拆成三个：

1. 在目的 $q$ 下，什么最终决策是好的？（→ Phase I）
2. 什么实验对达成该目的有价值？（→ Phase II-A）
3. RL 能否通过交互学会第 2 点？（→ 闭环 RL）

### 3.5 Phase I — 语言条件化的科学效用 ✅ STRONG GO

**问题**：模型能否只从自然语言目的判断哪个最终决策更好？（**不涉及实验设计**）

Bradley–Terry 偏好模型 $P_\phi(d_A \succ d_B \mid q, \omega) = \sigma(s_\phi(q,\omega,d_A) - s_\phi(q,\omega,d_B))$，六个目标族做 leave-one-family-out。

| 方法                  | IID            | 措辞 OOD       | 领域 OOD       | **held-out 目标族** | **Goal-flip** |
| --------------------- | -------------- | -------------- | -------------- | ------------------------- | ------------------- |
| B0 随机               | .500           | .500           | .500           | .500                      | .250                |
| B1 零样本 base        | .503           | .516           | .508           | .503                      | .000                |
| B3 无目的控制         | .760           | .748           | .731           | .512                      | .000                |
| B4 结构化 oracle      | .950           | .944           | .900           | .654                      | .749                |
| **B2 语言目的** | **.948** | **.941** | **.905** | **.708**            | **.732**      |

五个预注册假设全部通过（P1–P4 $p<10^{-4}$，P5 $p=0.017$）。关键诊断：无目的模型 IID .760 但 held-out 族塌到 .512，语言模型保持 .708（甚至高于结构化 oracle 的 .654）；只改目的、固定世界和候选决策，偏好能正确反转（.732）。

局限：held-out 假设检验族接近随机（.505），结构识别弱（.662），约束优化在硬约束违反时 regret 很大。

### 3.6 Phase II-A — Learned-DUG ❌ NO-GO

**问题**：把 Phase I 学到的效用 $s_\phi$ 冻结，用它估计实验的决策效用增益

$$
\widehat{\mathrm{DUG}}_\phi(e \mid q, D) = \mathbb{E}_y\big[\widehat V_\phi(q, D\cup\{e,y\})\big] - \widehat V_\phi(q, D)
$$

能否给出有用的实验排序？

中途做过一次数值可靠性修正：第一版 importance sampling 的 oracle 自身重复性就差（跨估计器一致性 ≈ 估计器自一致性，说明是方差不是偏差）。修正：软化 F5 惩罚的不连续、$n_\omega/n_y$ 提到 256/512、预注册排除自一致性 < 0.60 的 fold（保留 fold 自一致性 0.755–0.876，ESS 120–127/256；F1 识别 fold ≈ 0.529 被排除）。修正后：

| 假设                    | 效应       | 95% CI               | 结论                                       |
| ----------------------- | ---------- | -------------------- | ------------------------------------------ |
| P6 top-1 > 随机 (1/6)   | $+0.077$ | $[+0.053, +0.102]$ | 支持                                       |
| P7 regret < Generic-EIG | $-0.227$ | $[-0.386, -0.079]$ | **不支持**（8/8 可靠 fold 输给 EIG） |
| P8 排序 > 无目的排序    | $+0.029$ | $[-0.053, +0.110]$ | 不支持                                     |
| P9 gap recovery ≥ 0.30 | $-1.393$ | $[-1.924, -0.906]$ | 不支持（每个 fold 都为负）                 |

例：F3 预测族上 Generic-EIG Spearman 0.604 / regret 0.254，Learned-DUG 0.177 / 0.742。Transfer-A 与 Transfer-B 几乎无差别，说明失败不是"效用模型没见过该族"能解释的。

**结论**：**知道"什么结论符合目的" ≠ 知道"什么实验值得做"。** 按预注册，Learned-DUG 的 RL 阶段不启动。

### 3.7 Purpose-to-Reward — 把目的编译成终端 reward ✅ STRONG GO

**问题**：既然学到的效用不能当实验价值估计器，能否只把自然语言目的编译成一个**可执行的终端 reward 规范** $G_\phi(q) = \mathcal{R}_q$，中间的实验策略留给 RL？

$$
R_q(\tau) = U_q(d_T, \omega^*) - \sum_t c_q(e_t)
$$

$\mathcal{R}_q$ 定义任务成功、约束、实验成本，但**不**指定中间该选哪个实验。评估方式：在匹配的完整轨迹对 $(\tau_A, \tau_B)$ 上执行生成的 reward，看诱导的偏好是否与真实目的一致（行为评估，不看文本匹配）。

| 方法                  | IID             | 措辞 OOD       | **held-out 族** | 新目标+约束组合 | **Goal-flip** |
| --------------------- | --------------- | -------------- | --------------------- | --------------- | ------------------- |
| R0 随机 spec          | .559            | –             | .534                  | .585            | –                  |
| R1 零样本 base        | .504            | .515           | .501                  | .548            | .000                |
| R2 无目的控制         | .871            | .871           | .575                  | .734            | .000                |
| **R3 语言目的** | **1.000** | **.984** | **.875**        | **.885**  | **.823**      |
| R4 真值               | 1.000           | –             | 1.000                 | 1.000           | –                  |

五个预注册假设全部通过；成本敏感度**精确**（陈述成本 0.00 / 0.01 / 0.05 / 0.10 / 0.20 → 编译成本逐一相等，$\rho = 1.000$）。F5 约束优化最弱（.625），F6 假设检验 .815（Phase I 的失败项在这里反而成功）。

### 3.8 闭环 RL — 手工 reward vs 语言编译 reward

| 条件 | 含义                                                     |
| ---- | -------------------------------------------------------- |
| M0   | 仅 protocol SFT，无 RL                                   |
| M1   | Outcome RL：只评任务结果，**故意去掉**约束和成本项 |
| M2   | Manual-spec RL：用人工写的真值 reward                    |
| M3   | 语言编译 reward RL：用 frozen Purpose-to-Reward 编译     |

主指标：ID 终端效用 regret（greedy 解码，一律用真值 reward 打分，oracle regret = 0）。$n=200$ / split，**5 个真正独立的 replicate**（训练/测试世界和 SFT 数据都随 replicate 变化），4 × 5 = 20 次训练。

> 注：更早一轮探索性结果（"M3 ≈ M2 且在 F5 上优于 M1"）被 **train-mode 评估 bug** 作废，不作为证据；旧输出保留在 `results_trainmode_bug` 下未覆盖。因此增加了 answer-rate gate（H4）并做独立确认性复现。

**结果**：

|                                     | M0             | M1                       | M2             | M3             |
| ----------------------------------- | -------------- | ------------------------ | -------------- | -------------- |
| ID regret（5 replicate 均值 ± SD） | 1.236 ± 0.114 | **1.001** ± 0.108 | 1.020 ± 0.106 | 1.012 ± 0.106 |

- **H1**（非劣性，$\mathrm{regret}(M3) - \mathrm{regret}(M2) < 0.10$，边界 = 早期探索效应 0.408 的约 1/4）：5 个 replicate 差值 $+0.005, -0.076, 0.000, +0.057, -0.027$；一个 replicate 条件塌缩不可用，其余 4 个全部在边界内（2 正 2 负；5 个均值 $-0.011$，SD $0.056$）→ **支持**
- **H2**（$\mathrm{regret}(M3) < \mathrm{regret}(M0)$）：5/5 → 支持
- **H3**（held-out 约束族 F5 上 $\mathrm{regret}(M1) > \mathrm{regret}(M2)$，因为 M1 没有约束/成本项）：2 支持 2 反对 → **不复现**
- **H4**（answer rate = 1.0）：通过

**正确的解读很窄**：在当前 pipeline 里，语言编译的 reward 非劣于手工 reward。H1 **不**说明 M2 或 M3 学会了做实验。

### 3.9 行为塌缩诊断 ⚠️

对全部 20 个 adapter 在数值任务族上做事后诊断（$n=100$）：

- 预测值与真值的相关系数：**$-0.089 \sim 0.241$**
- 数值 regret：$1.07 \sim 1.37$
- 对照：**输出先验均值 regret = 1.144，永远输出 0 = 1.115**，不回答 = 10，oracle = 0
- 一个 replicate 的多个 RL 条件**完全塌缩**：$\mathrm{frac}(\hat d = 0) = 1$，$\mathrm{SD}(\hat d) = 0$；其余 replicate 多为部分塌缩或接近先验

也就是说，RL 后的模型在数值任务上和"输出常数"几乎没区别。H2 的 aggregate 提升主要是学到了强先验 / 中心趋势捷径，不是利用证据。H3 的失败说明即使 reward 规范语义正确，RL 也没学会利用其中的约束和成本项。

$$
\boxed{q \to R_q \ \text{已经可行}} \qquad\qquad \boxed{R_q \to \text{证据依赖的实验行为 尚未证明}}
$$

## 4. 目前结果怎么样

### 状态表

| 阶段              | 问题                                             | 状态                             |
| ----------------- | ------------------------------------------------ | -------------------------------- |
| 因果 pilot        | epistemic RL 能否改进窄任务的实验选择？          | 部分正面（需 SFT）               |
| 机制消融          | step credit / token mask 是关键吗？              | 否                               |
| Round 2           | SFT 初始化影响 epistemic RL 可学性？             | 是；倒 U，oracle-action SFT 最强 |
| Phase I           | 语言能否指定新目的下的好决策？                   | **STRONG GO**              |
| Phase II-A        | 学到的效用能否转成实验价值？                     | **NO-GO**                  |
| Purpose-to-Reward | 语言能否编译成可执行终端 reward？                | **STRONG GO**              |
| 闭环 H1           | 编译 reward 非劣于手工 reward？                  | 支持                             |
| 闭环 H2           | 编译 reward RL 优于 protocol SFT？               | 支持，但行为上被削弱             |
| 闭环 H3           | 约束/成本项在 held-out 约束任务上有用？          | 不支持                           |
| 行为能力          | RL agent 学会了证据依赖的实验？                  | **未建立**                 |
| 下一步 audit      | 失败在推断、证据使用、选择、适应还是 benchmark？ | 待做                             |

### 已验证

1. 窄因果任务上，SFT + epistemic RL 优于 SFT + outcome RL
2. 该任务上细粒度信用分配不是主要机制
3. 科学目的的语义可学习，能跨 held-out 目标族迁移、随 goal-flip 正确反转
4. 自然语言目的可编译成可执行终端 reward（含方向、约束、成本强度），held-out 泛化强
5. 编译 reward 在当前 RL pipeline 中可替代手工 reward（预注册 0.10 边界内非劣）

### 已否定

1. 静态效用 ⇒ 实验价值（Learned-DUG 输给 Generic-EIG）
2. 仅理解后验 / 状态就能解释 epistemic RL 增益（Reasoning-SFT ≪ Oracle-action SFT）
3. step / token 信用修复能解决因果 pilot
4. 当前 aggregate RL 提升 = 学会了科学实验

### 未解决（核心瓶颈）

> **为什么 LLM 能优化一个正确的科学 reward，却没有学会利用实验证据？**

候选失败位置：(i) 即使证据质量高也无法从中推断；(ii) 实验选择失败；(iii) 无法根据不同结果调整后续实验；(iv) reward / benchmark 存在捷径，使先验式回答有竞争力。

报告提出的下一步是一个**行为审计（audit）**，用最干净的 M2 reward、在新生成的诊断世界上评估已有 M0 / M2 模型，主看 F2 / F3 / F4。四个测试：

- **A 证据利用**：同一 episode 的 $y_1$ 换成 true / shuffled（同族同实验同量级的别的世界的结果）/ masked 三种，看 $\Delta_{\mathrm{shuffle}} = \mathrm{Regret}_{\mathrm{shuffle}} - \mathrm{Regret}_{\mathrm{true}} > 0$ 是否成立
- **B oracle 证据定位**：拿掉实验选择责任，B0 不做实验 / B1 随机轨迹 / B2 oracle 开环轨迹 / B3 oracle 自适应轨迹，LLM 只做最终推断。B3 ≈ B0 → 推断瓶颈；B3 ≪ B0 但 M2 ≫ B3 → 采集瓶颈
- **C 自适应分支**：固定 $(q, D_0, e_1)$，采样不同 $y^{(m)}$，只保留最优后续实验不同且价值差 > $\delta$ 的分支对，看模型是否选出不同**且正确**的下一步（NOB 指标）
- **D 捷径基线**：$\pi_{\mathrm{prior}}, \pi_0, \pi_{\mathrm{noexp}}, \pi_{\mathrm{fixed}}, \pi_{\mathrm{random}}, \pi_{\mathrm{oracle}}$，定义 ScientificGain = $J(M2) - \max\{\text{trivial}\}$

并预先划定两个子集（experiment-needed、evidence-matters），按结果走四分支决策树。若定位为序列信用问题，再上**基于后果的 counterfactual RL**（对每个候选实验 rollout $M$ 条完整后续、用终端 reward 估 $\widehat Q^{\pi_\theta}$、group-relative advantage、含 `STOP` 动作）——报告明确把它列为条件性下一步，不是已验证贡献。

## 5. 读后分析（整理者观点，非报告原文）

这些是我读完后觉得最值得警惕、但报告里没有充分展开的点。

**5.1 M1 是四个条件里最好的，这比 H3 失败更值得注意。**
M1 故意去掉约束和成本项，但用**含约束和成本的真值 reward** 打分后，它的 regret（1.001）仍然是最低的。这意味着 reward 里的约束/成本项对训练**完全没有产生影响**——不是"学得不够好"，是"没有梯度"。最机械的可能：模型的实验次数 $N_{\mathrm{exp}}$ 根本没在变（总是打满 budget，或总是立刻停），那么 $\lambda N_{\mathrm{exp}}$ 是常数，在 group-relative advantage 里被减掉了。报告全文**没有报告任何条件下的实验次数分布**，这是最便宜也最该先补的诊断。

**5.2 reward landscape 在先验附近是平的。**
数值族上："永远输出 0" 1.115、先验均值 1.144、训练后 1.07–1.37、oracle 0。trivial 策略到最好的训练模型不到 0.1，训练模型到 oracle 约 1.0。RL 只需要学会输出常数就能拿到几乎全部"容易的"收益，剩下 90% 的 gap 需要真正做实验才能拿到——但那部分的信号淹没在 GRPO 组内噪声里。这对应决策树第 4 条（重新设计 benchmark），我认为这条的先验概率并不低，不应放在最后。

**5.3 Round 2 的结论对论文叙事是双刃剑。**
Oracle-action SFT 带来最大的 epistemic RL 增益，Reasoning-SFT 几乎没有。这说明 RL 收益很大程度上建立在"先见过 oracle 怎么选实验"上。虽然 oracle 只在训练时使用，不违反部署 frozen 的要求，但 reviewer 会问：这是"学会了实验"还是"蒸馏 + 微调"？需要一条**没有 oracle-action 暴露**也能起效的路径，或者坦白把 oracle-action SFT 当作方法的一部分。

**5.4 因果 pilot 里 1.8 万参数的 MLP 打败所有 LLM。**
这是个必须正面回答的问题：为什么要用 LLM？答案应该是"语言接口 + 跨目的泛化"（Phase I 和 P2R 支持这一点），但目前还没有一个实验同时展示"LLM 在多目的设置下做实验 > 专用小模型"。

**5.5 Purpose-to-Reward 的 IID 1.000 是天花板效应。**
随机 spec 都能拿 .559、无目的控制 .871，说明轨迹对评估里很多 pair 是任何合理 reward 都能分开的。真正的证据是 held-out 族 .875 vs .575 和 goal-flip .823 vs .000 这两列。写论文时应以这两列为主打。

**5.6 H1 的统计功效有限。**
只有 4 个 informative replicate，差值 SD 0.056，边界 0.10。结论成立，但如果 M3 真的比 M2 差 0.05，这个设计大概率也检测不出来。作为"非劣性"结论是够的，不要过度解读成"等价"。

**5.7 H2 的提升到底来自哪个目标族？**
aggregate regret 从 1.236 降到 ~1.0，但数值族上所有 adapter 都接近先验。那 0.23 的下降是数值族从"乱输出"变成"输出先验均值"，还是离散族（F1 / F6）真的有进步？报告没有分族报告闭环结果。如果离散族有真实进步，那是目前唯一可能存在的"真行为"证据，值得单独挖。

## 6. 建议

### 6.1 audit 之前，先做三个便宜的诊断（一天内能出）

1. **每个条件的 $N_{\mathrm{exp}}$ 分布和 STOP 时机**。如果 M0–M3 的实验次数分布几乎一样，5.1 的假设成立，H3 失败和 M1 最优就都解释了，audit 里关于成本的部分可以省掉。
2. **闭环结果分目标族拆开**，每族同时报 $\pi_{\mathrm{prior}}$ 和 $\pi_0$ 的 regret。看 H2 的提升到底落在哪。
3. **reward 各项的量级**：训练分布上 $U_q$ 的组内标准差 vs $\lambda N_{\mathrm{exp}}$ 的组内标准差 vs $\eta[g]_+$ 的非零比例。如果成本项的方差比效用项小一两个数量级，它在 advantage 里就是噪声。

### 6.2 对 audit 设计的修改建议

- **先跑 B（oracle 证据定位）**，它是决策树的分叉点，A / C / D 的解读都依赖它。
- 给 B3 加一个**同证据下的 Bayes 最优推断**作为参照：$\mathrm{Regret}_{B3}(\mathrm{LLM}) - \mathrm{Regret}_{B3}(\mathrm{Bayes})$ 直接量化推断差距，比只和 B0 比更干净。
- 把 **experiment-needed subset 作为主指标**而不是子集。在 $\mathrm{Regret}_{\mathrm{prior}}(\omega) - \mathrm{Regret}_{\mathrm{oracle}}(\omega) < \gamma$ 的世界上，做不做实验本来就无所谓，混进来只会稀释信号。
- 测试 A 的 masked 条件是天然下界，报告时给出三元排序 true / shuffled / masked，期望 true 明显优于后两者且后两者相近。如果 shuffled 比 masked 差很多，说明模型在"用"证据但用错了，这是另一种诊断。
- 决策树里的"≈"和"≪"要**预先定义阈值**，否则事后又是一轮解释权争夺。

### 6.3 audit 之后的分支

**若定位为推断失败**（oracle 证据也帮不了）：

- 先做**证据整合 SFT**：监督目标为 $(q, h_t) \to$ Bayes 最优决策 / 后验，模拟器能廉价生成无限量。Round 2 已经证明"对的 SFT"比 RL 花样重要得多。
- 或者**给模型一个拟合工具**（calculator / curve-fit / posterior-update tool），把"数值推断"和"实验策略"解耦。真实科学家也是用软件拟合的，LLM 的核心任务应该是策略。这同时消除了数值捷径的混淆。这是正当的设计选择，不是作弊——但需要在论文里重新界定 claim。

**若定位为选择 / 适应失败**（oracle 证据有用，但模型自己采集的证据没用）：

- 报告的 counterfactual RL 方向合理，但成本是每步 $K \times M$ 条 LLM rollout。建议：结果采样 $y^{(k,m)}$ 用模拟器（便宜），只在 LLM 续写上花预算；先 $K=3, M=4$ 起步。
- 一个更便宜的替代：训练时用 **oracle DUG 做 privileged dense shaping**（部署时不需要）。Round 2 说明 oracle 信息在训练时有效；Phase II-A 否定的是 *learned* DUG，不是 oracle DUG。注意 Phase II-A 已暴露 oracle DUG 估计本身的方差问题（自一致性 0.53–0.88），作为 dense signal 需要更多样本。

**若定位为 benchmark 捷径**（trivial 策略接近 M2）：

- 见 6.4。这种情况下继续调 RL 是浪费算力。

### 6.4 指标与 benchmark

- 把 headline 指标改成 **gap recovery**：$\dfrac{\mathrm{Regret}_{\mathrm{prior}} - \mathrm{Regret}_{\mathrm{model}}}{\mathrm{Regret}_{\mathrm{prior}} - \mathrm{Regret}_{\mathrm{oracle}}}$，0 = 和不做实验一样，1 = oracle。Phase II-A 已经用过这个概念，全项目统一。按数值族现在的数字粗算，最好的 adapter 也不到 0.1，有的为负。
- 生成世界时**筛掉先验就能答好的世界**，让"必须做实验"成为常态而不是子集。
- 考虑在 reward 里给**证据一致性**一个显式项（最终预测对观测数据的似然），否则模型没有理由看 $y_t$。这算改 reward，报告主张 audit 前不动 reward，所以列为 audit 后的选项。

### 6.5 论文写作

- **可以现在就写的故事**：Phase I + Purpose-to-Reward + 闭环非劣性 + 诚实的负面行为诊断 = "语言作为科学实验 RL 的 reward 规范接口：什么可行、什么还不行"。两个 STRONG GO 加一个预注册、多重校正过的负面结果，比硬凑一个正面闭环结果更可信。
- **reviewer 一定会攻的点**（对应 5.x）：MLP > LLM 为何还用 LLM；oracle-action SFT 依赖；M1 最优 / 约束成本项无效；P2R IID 天花板；H1 只有 4 个 replicate；全是模拟环境。每条都要在 limitation 里先说。
- **报告目前缺的实验细节**：base model 和参数量、RL 算法（GRPO？）与超参、训练步数、实验 budget $T$、动作空间大小、每阶段的 $n$、六个目标族的具体环境定义。写论文前需要全部补齐，最好现在就从训练配置里导出。
- tex 里 `\author{}` 是空的；两个 `.bib` 文件目前只是 Overleaf 链接占位符。

### 6.6 仓库

- `result-tex/` 目前是 untracked，建议 commit。
- 仓库里目前**只有 tex，没有代码、配置和原始结果**（报告引用的 `results_trainmode_bug` 等不在库里）。至少把环境定义、reward 规范、训练配置、评估脚本纳入版本控制，否则 audit 无法复现。
- `.gitignore` 里的 `paper/*` 目录不存在，可能是历史遗留。

## 7. 仓库结构与编译

```
llm-bio-experiment-agent/
├── README.md                     ← 本文件
├── .gitignore
└── result-tex/
    ├── scientific_experimentation_rl_full_report.tex   ← 完整实验记录（719 行）
    ├── scientific_experimentation_rl_full_report.pdf   ← 15 页
    ├── proposal.bib / proj_ref.bib                     ← 占位（Overleaf 链接）
    └── fancyhdr.sty
```

编译：

```bash
cd result-tex && latexmk -pdf scientific_experimentation_rl_full_report.tex
```
