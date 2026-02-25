# 迁移到自回归发育模型

## 当前架构的问题

### 现有：条件Flow Matching
```
输入: t, x_t (随机噪声或中间状态)
输出: v(t, x_t) = 指向x_1的速度
损失: ||x_t + (1-t)*v - x_1||^2
问题: 细胞数是固定的，分裂是预先确定的
```

### 目标：自回归动力学
```
输入: x_t (当前状态)
输出: (x_{t+dt}, split_probs, del_probs)
损失: -log p(x_{t+dt} | x_t) + 事件预测交叉熵
优势: 动态细胞数，真正的因果模拟
```

---

## 可迁移组件 (Reuse)

### 1. 状态表示 ✅
```python
# src/branching_flows/states.py - 完全复用
@dataclass
class BranchingState:
    states: tuple[torch.Tensor, ...]  # (continuous, discrete)
    groupings: torch.Tensor
    padmask: torch.Tensor
    flowmask: torch.Tensor
    branchmask: torch.Tensor
    del_flags: torch.Tensor
    ids: torch.Tensor
```

### 2. 基因/空间双流编码 ✅
```python
# crossmodal_model.py 中的投影层
gene_proj: Linear(2000, d_model//2)
spatial_proj: Linear(3, d_model//2)
```

### 3. 交叉模态注意力 ✅
```python
# CrossModalFusion 层完全复用
CrossModalFusion(d_model, n_heads)
```

### 4. 预测头结构 ✅
```python
# 头的定义不变，但使用方式改变
gene_head: Linear(d_model, 2000)      # 输出变为"delta"而非绝对值
spatial_head: Linear(d_model, 3)       # 输出速度
split_head: Linear(d_model, 1)         # 分裂概率（已有）
del_head: Linear(d_model, 1)           # 删除概率（已有）
```

---

## 需要重新设计的组件

### 1. 核心前向传播 ⚠️ NEW
```python
class AutoregressiveNemaModel(nn.Module):
    def forward_step(self, state: BranchingState) -> StepOutput:
        """单步演化：输入x_t，输出(x_{t+dt}, events)"""
        # 1. 编码当前状态
        gene_emb = self.gene_proj(state.states[0][..., :2000])
        spatial_emb = self.spatial_proj(state.states[0][..., -3:])

        # 2. 交叉模态融合（复用）
        fused = self.cross_modal_fusion(gene_emb, spatial_emb)

        # 3. 预测变化量（而非绝对值）
        gene_delta = self.gene_head(fused) * dt  # 基因表达变化
        spatial_vel = self.spatial_head(fused) * dt  # 空间速度
        split_logits = self.split_head(fused)  # 分裂概率
        del_logits = self.del_head(fused)      # 删除概率

        # 4. 事件采样（随机性或确定性）
        split_probs = torch.sigmoid(split_logits)
        del_probs = torch.sigmoid(del_logits)

        return StepOutput(
            gene_delta=gene_delta,
            spatial_vel=spatial_vel,
            split_probs=split_probs,
            del_probs=del_probs,
        )
```

### 2. 动态序列管理 ⚠️ NEW
```python
class DynamicCellManager:
    """管理细胞的分裂和删除，动态调整序列长度"""

    def apply_events(
        self,
        state: BranchingState,
        split_probs: torch.Tensor,
        del_probs: torch.Tensor,
        split_threshold: float = 0.5,
        del_threshold: float = 0.5,
    ) -> BranchingState:
        """
        根据概率执行事件：
        - split_prob > threshold: 1 cell → 2 cells
        - del_prob > threshold: remove cell
        """
        # 实现动态扩缩容
        pass
```

### 3. 训练目标 ⚠️ NEW
```python
# 不再是flow matching，而是下一状态预测
def autoregressive_loss(
    pred_delta: torch.Tensor,      # 预测的变化
    true_delta: torch.Tensor,      # 真实变化
    pred_split: torch.Tensor,      # 预测分裂概率
    true_split: torch.Tensor,      # 真实是否分裂
    pred_del: torch.Tensor,        # 预测删除概率
    true_del: torch.Tensor,        # 真实是否删除
) -> torch.Tensor:

    # 状态变化MSE
    state_loss = F.mse_loss(pred_delta, true_delta)

    # 事件预测BCE
    split_loss = F.binary_cross_entropy_with_logits(pred_split, true_split)
    del_loss = F.binary_cross_entropy_with_logits(pred_del, true_del)

    return state_loss + lambda_split * split_loss + lambda_del * del_loss
```

### 4. 数据加载 ⚠️ NEW
```python
# 当前：采样单点时间片
sample = dataset[i]  # 某个时间bin的细胞

# 新：采样轨迹片段
trajectory = trajectory_dataset[i]  # [(t0, x0), (t1, x1), ..., (tn, xn)]
# 需要连续时间序列，而非离散bins
```

### 5. 训练循环 ⚠️ NEW
```python
# 当前：随机时间t，条件生成
for batch in loader:
    t = torch.rand(B)
    loss = flow_matching_loss(model, batch, t)

# 新：沿轨迹展开多步
for trajectory in loader:
    total_loss = 0
    for t in range(len(trajectory)-1):
        pred = model.step(trajectory[t])
        loss = autoregressive_loss(pred, trajectory[t+1])
        total_loss += loss
```

---

## 实现路线图

### Phase 1: 核心架构 (2-3天)
1. `AutoregressiveNemaModel`
   - 复用CrossModalNemaModel的编码器
   - 新的`forward_step`方法
   - 输出变化量而非绝对值

2. `DynamicCellManager`
   - 处理分裂/删除事件
   - 动态batching

### Phase 2: 数据管道 (1-2天)
1. `TrajectoryDataset`
   - 从WormGUIDES提取连续时间轨迹
   - 或从Sulston树生成合成轨迹

2. 轨迹augmentation
   - 子采样
   - 时间缩放

### Phase 3: 训练 (3-5天)
1. 新训练循环
2. 调试稳定性（梯度爆炸/消失）
3. 超参数调优

### Phase 4: 验证 (2-3天)
1. 从单细胞开始生成完整胚胎
2. 扰动实验（真正的因果测试）
3. 与真实Sulston树对比

---

## 技术挑战

### 挑战1: 动态序列长度
**问题**: PyTorch需要固定tensor shape
**方案**:
- 使用max_cells + padding mask
- 或分批次处理不同长度的序列

### 挑战2: 分裂事件离散性
**问题**: 分裂是离散的（0/1），但梯度需要连续
**方案**:
- Gumbel-Softmax重参数化
- 或straight-through estimator

### 挑战3: 长程依赖
**问题**: 单细胞→1000细胞需要1000步，梯度消失
**方案**:
- 截断BPTT (Truncated Backpropagation Through Time)
- 分层训练（先学短序列，再学长序列）

---

## 迁移总结

| 组件 | 复用度 | 工作量 | 文件 |
|------|--------|--------|------|
| 状态表示 | 100% | 0 | states.py |
| 编码器 | 90% | 小 | crossmodal_model.py |
| 预测头 | 70% | 中 | 改输出为delta |
| 前向传播 | 10% | 大 | 新文件 |
| 数据加载 | 20% | 大 | 新文件 |
| 训练循环 | 5% | 大 | 新训练脚本 |

**总估算**: 2-3周实现+训练

---

## 立即可做的事

1. **创建新模型文件**: `src/branching_flows/autoregressive_model.py`
2. **设计轨迹数据集**: 从现有数据构建连续时间序列
3. **实现单步演化**: 先不处理动态长度，固定max_cells
4. **简单训练测试**: 用短轨迹（10→20细胞）验证可行性

要我立即开始实现Phase 1吗？
