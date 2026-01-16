# CShaper 数据整合方案

## 概述

本文档详细规划如何将 CShaper 4D形态学图谱数据整合到 NemaContext 的三模态框架中。

## 1. 当前架构分析

### 1.1 数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         当前数据处理管线                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Large2025/Packer2019          WormGUIDES           WormBase          │
│   (scRNA-seq MTX)               (nuclei files)       (lineage JSON)    │
│         │                            │                    │            │
│         ▼                            ▼                    ▼            │
│   ┌────────────────┐         ┌────────────────┐   ┌────────────────┐   │
│   │ExpressionLoader│         │ SpatialMatcher │   │ LineageEncoder │   │
│   │                │         │                │   │                │   │
│   │ - load_large2025()       │ - match_by_time()  │ - parse_lineage()  │
│   │ - load_packer2019()      │ - match_by_lineage()│ - encode_binary() │
│   │ - CSR matrix output      │ - XYZ coords   │   │ - build_adjacency()│
│   └────────────────┘         └────────────────┘   └────────────────┘   │
│         │                            │                    │            │
│         └────────────────────────────┼────────────────────┘            │
│                                      │                                 │
│                                      ▼                                 │
│                          ┌────────────────────────┐                    │
│                          │TrimodalAnnDataBuilder  │                    │
│                          │                        │                    │
│                          │ - build()              │                    │
│                          │ - build_spatial_graph()│                    │
│                          │ - build_lineage_graph()│                    │
│                          └────────────────────────┘                    │
│                                      │                                 │
│                                      ▼                                 │
│                          ┌────────────────────────┐                    │
│                          │     AnnData Output     │                    │
│                          │                        │                    │
│                          │ X: expression matrix   │                    │
│                          │ obsm['X_spatial']      │                    │
│                          │ obsm['X_lineage_binary']│                   │
│                          │ obsp['spatial_distances']│                  │
│                          │ obsp['lineage_adjacency']│                  │
│                          └────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

| 组件 | 文件 | 职责 |
|-----|-----|-----|
| `ExpressionLoader` | `src/data/builder/expression_loader.py` | 加载Large2025/Packer2019表达矩阵 |
| `SpatialMatcher` | `src/data/builder/spatial_matcher.py` | 匹配WormGUIDES空间坐标 |
| `LineageEncoder` | `src/data/builder/lineage_encoder.py` | 解析/编码谱系名称 |
| `WormAtlasMapper` | `src/data/builder/worm_atlas.py` | 细胞类型↔谱系映射 |
| `TrimodalAnnDataBuilder` | `src/data/builder/anndata_builder.py` | 整合构建AnnData |
| `SpatialGraphBuilder` | `src/model/spatial_graph.py` | 构建空间邻居图 (k-NN) |

### 1.3 当前局限性

1. **空间数据仅为点坐标**: WormGUIDES只提供核心位置，无细胞形态
2. **邻居图基于k-NN近似**: 非真实物理接触
3. **缺少形态学特征**: 无体积、表面积、不规则度等
4. **时间覆盖有限**: Large2025时间分布与WormGUIDES不完全对齐

---

## 2. CShaper 数据资源

### 2.1 可用数据

| 数据文件 | 路径 | 内容 | 关键字段 |
|---------|-----|-----|---------|
| **ContactInterface/** | `dataset/raw/cshaper/ContactInterface/Sample*_Stat.csv` | 细胞-细胞接触面积矩阵 | 对称矩阵，非零值=接触面积(μm²) |
| **VolumeAndSurface/** | `dataset/raw/cshaper/VolumeAndSurface/Sample*_Stat.csv` | 细胞体积/表面积时间序列 | 每行=时间帧，每列=细胞 |
| **Standard Dataset 1** | `dataset/raw/cshaper/Standard Dataset 1/*.mat` | 标准化3D坐标 (46胚胎平均) | 按谱系树结构组织 (gen×pos) |
| **Standard Dataset 2** | `dataset/raw/cshaper/Standard Dataset 2/*.mat` | 3D体素分割结果 | 184×114×256 矩阵 |

### 2.2 数据特性

```
CShaper 时间范围: 4-350细胞期 (约20-380分钟)
时间帧数: 54帧
胚胎数: 17个(完整膜分割) + 29个(仅核追踪) = 46个标准数据集
细胞命名: 标准C. elegans谱系命名 (ABa, ABp, MS, E, C, ...)
```

---

## 3. 整合方案

### 3.1 新组件设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         增强后的数据处理管线                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Large2025          WormGUIDES       WormBase        CShaper          │
│   (scRNA-seq)        (nuclei)         (lineage)       (morphology)     │
│        │                 │                │               │            │
│        ▼                 ▼                ▼               ▼            │
│   ┌──────────┐     ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│   │Expression│     │ Spatial  │    │ Lineage  │    │ CShaper      │   │
│   │ Loader   │     │ Matcher  │    │ Encoder  │    │ Processor    │ NEW
│   └──────────┘     └──────────┘    └──────────┘    │              │   │
│        │                 │                │        │-load_contact()│   │
│        │                 │                │        │-load_volume() │   │
│        │                 │                │        │-load_spatial()│   │
│        │                 │                │        └──────────────┘   │
│        │                 │                │               │            │
│        └─────────────────┴────────────────┴───────────────┘            │
│                                   │                                    │
│                                   ▼                                    │
│                     ┌──────────────────────────────┐                   │
│                     │  EnhancedAnnDataBuilder      │ NEW (extends)     │
│                     │                              │                   │
│                     │  - build_with_cshaper()      │                   │
│                     │  - _add_morphology()         │                   │
│                     │  - _add_contact_graph()      │                   │
│                     │  - _enhance_spatial()        │                   │
│                     └──────────────────────────────┘                   │
│                                   │                                    │
│                                   ▼                                    │
│                     ┌──────────────────────────────┐                   │
│                     │     Enhanced AnnData         │                   │
│                     │                              │                   │
│                     │  X: expression matrix        │                   │
│                     │                              │                   │
│                     │  obsm['X_spatial']           │ (improved)        │
│                     │  obsm['X_lineage_binary']    │                   │
│                     │  obsm['X_cshaper_spatial']   │ NEW               │
│                     │                              │                   │
│                     │  obs['cell_volume']          │ NEW               │
│                     │  obs['cell_surface']         │ NEW               │
│                     │  obs['sphericity']           │ NEW               │
│                     │                              │                   │
│                     │  obsp['contact_adjacency']   │ NEW (真实接触)    │
│                     │  obsp['spatial_distances']   │                   │
│                     │  obsp['lineage_adjacency']   │                   │
│                     └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 新增文件

```
src/data/
├── builder/
│   ├── cshaper_processor.py    # NEW: CShaper数据处理器
│   └── enhanced_builder.py     # NEW: 增强版AnnData构建器
├── processors/
│   └── cshaper/               # NEW: CShaper专用处理
│       ├── __init__.py
│       ├── contact_loader.py   # 接触矩阵加载
│       ├── morphology_loader.py# 形态特征加载
│       └── spatial_loader.py   # Standard Dataset空间数据
```

---

## 4. 详细实现计划

### 4.1 Phase 1: CShaper处理器 (`src/data/builder/cshaper_processor.py`)

```python
class CShaperProcessor:
    """
    CShaper数据处理器 - 加载和处理形态学数据
    """
    
    def __init__(self, data_dir: str = "dataset/raw"):
        self.data_dir = Path(data_dir)
        self.cshaper_dir = self.data_dir / "cshaper"
        
    # === 接触矩阵 ===
    def load_contact_matrices(self, samples: List[str] = None) -> Dict[str, pd.DataFrame]:
        """加载所有样本的细胞-细胞接触矩阵"""
        
    def get_contact_adjacency(self, lineage_names: List[str], 
                               time_frame: int = None) -> csr_matrix:
        """为给定细胞列表构建接触邻接矩阵"""
        
    def get_average_contacts(self) -> pd.DataFrame:
        """跨样本平均的接触强度"""
    
    # === 形态特征 ===    
    def load_volume_surface(self, samples: List[str] = None) -> Dict[str, pd.DataFrame]:
        """加载体积/表面积时间序列"""
        
    def get_morphology_features(self, lineage_names: List[str],
                                 time_frame: int = None) -> pd.DataFrame:
        """获取细胞形态特征 (volume, surface, sphericity)"""
    
    # === 标准化空间坐标 ===
    def load_standard_spatial(self) -> Dict[str, np.ndarray]:
        """加载Standard Dataset 1的标准化坐标"""
        
    def get_spatial_coords(self, lineage_names: List[str],
                           time_frame: int = None) -> np.ndarray:
        """获取标准化3D坐标"""
    
    # === 时间映射 ===
    def embryo_time_to_frame(self, time_min: float) -> int:
        """将胚胎时间(分钟)映射到CShaper帧号"""
        
    def frame_to_embryo_time(self, frame: int) -> float:
        """将CShaper帧号映射到胚胎时间(分钟)"""
```

### 4.2 Phase 2: 增强版构建器 (`src/data/builder/enhanced_builder.py`)

```python
class EnhancedAnnDataBuilder(TrimodalAnnDataBuilder):
    """
    增强版三模态AnnData构建器 - 集成CShaper数据
    """
    
    def __init__(self, data_dir: str = "dataset/raw", 
                 output_dir: str = "dataset/processed"):
        super().__init__(data_dir, output_dir)
        self.cshaper = CShaperProcessor(data_dir)
    
    def build_with_cshaper(
        self,
        variant: Literal["complete", "extended"] = "complete",
        source: Literal["large2025", "packer2019"] = "large2025",
        include_morphology: bool = True,
        include_contact_graph: bool = True,
        use_cshaper_spatial: bool = False,  # 是否用CShaper替换WormGUIDES
        **kwargs
    ) -> ad.AnnData:
        """构建包含CShaper增强的AnnData"""
        
        # 1. 先构建基础三模态
        adata = self.build(variant=variant, source=source, **kwargs)
        
        # 2. 添加CShaper增强
        if include_morphology:
            adata = self._add_morphology_features(adata)
            
        if include_contact_graph:
            adata = self._add_contact_graph(adata)
            
        if use_cshaper_spatial:
            adata = self._enhance_spatial_coords(adata)
            
        return adata
    
    def _add_morphology_features(self, adata: ad.AnnData) -> ad.AnnData:
        """添加细胞体积、表面积、球形度"""
        
        # 获取谱系名称和时间
        lineages = adata.obs['lineage_complete'].values
        times = adata.obs.get('embryo_time_min', None)
        
        # 匹配CShaper形态数据
        morphology = self.cshaper.get_morphology_features(lineages, times)
        
        # 添加到obs
        adata.obs['cell_volume'] = morphology['volume'].values
        adata.obs['cell_surface'] = morphology['surface'].values
        adata.obs['sphericity'] = morphology['sphericity'].values
        adata.obs['has_morphology'] = ~morphology['volume'].isna()
        
        return adata
    
    def _add_contact_graph(self, adata: ad.AnnData) -> ad.AnnData:
        """添加真实细胞-细胞接触图"""
        
        lineages = adata.obs['lineage_complete'].values.tolist()
        
        # 构建接触邻接矩阵
        contact_adj = self.cshaper.get_contact_adjacency(lineages)
        
        # 添加到obsp
        adata.obsp['contact_adjacency'] = contact_adj
        adata.obsp['contact_distances'] = self._contact_to_distance(contact_adj)
        
        # 统计
        n_edges = contact_adj.nnz // 2
        logger.info(f"Built contact graph with {n_edges} edges")
        
        return adata
    
    def _enhance_spatial_coords(self, adata: ad.AnnData) -> ad.AnnData:
        """使用CShaper标准化坐标增强/替换WormGUIDES坐标"""
        
        lineages = adata.obs['lineage_complete'].values.tolist()
        times = adata.obs.get('embryo_time_min', None)
        
        # 获取CShaper坐标
        cshaper_coords = self.cshaper.get_spatial_coords(lineages, times)
        
        # 保存为额外的obsm (保留原始WormGUIDES)
        adata.obsm['X_cshaper_spatial'] = cshaper_coords
        
        return adata
```

### 4.3 Phase 3: 接触图加载器 (`src/data/processors/cshaper/contact_loader.py`)

```python
class ContactLoader:
    """
    CShaper ContactInterface数据加载器
    
    文件格式: Sample{04-20}_Stat.csv
    - 行/列名: 细胞谱系名称 (ABa, ABp, MS, E, ...)
    - 值: 接触面积 (μm²), 0表示无接触
    - 矩阵对称
    """
    
    def __init__(self, contact_dir: Path):
        self.contact_dir = contact_dir
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_sample(self, sample_id: str) -> pd.DataFrame:
        """加载单个样本的接触矩阵"""
        
    def load_all_samples(self) -> Dict[str, pd.DataFrame]:
        """加载所有样本"""
        
    def get_consensus_contacts(self, 
                                min_samples: int = 3) -> pd.DataFrame:
        """获取共识接触 (在>=min_samples个样本中出现)"""
        
    def get_contact_strength(self, 
                             cell1: str, 
                             cell2: str,
                             sample_id: str = None) -> float:
        """获取两个细胞间的接触强度"""
        
    def build_sparse_adjacency(self, 
                               cell_list: List[str],
                               sample_id: str = None,
                               threshold: float = 0.0) -> csr_matrix:
        """为给定细胞列表构建稀疏邻接矩阵"""
```

### 4.4 Phase 4: 形态特征加载器 (`src/data/processors/cshaper/morphology_loader.py`)

```python
class MorphologyLoader:
    """
    CShaper VolumeAndSurface数据加载器
    
    文件格式: Sample{04-20}_Stat.csv
    - 行: 时间帧 (0-53)
    - 列: 细胞谱系名称
    - 值: 体积(列名含volume)或表面积(列名含surface)
    """
    
    def __init__(self, volume_dir: Path):
        self.volume_dir = volume_dir
        
    def load_sample(self, sample_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载单个样本的体积和表面积数据"""
        
    def get_cell_stats(self, 
                       cell_name: str,
                       sample_id: str = None) -> Dict[str, np.ndarray]:
        """获取单个细胞的体积/表面积时间序列"""
        
    def get_morphology_at_time(self,
                               time_frame: int,
                               sample_id: str = None) -> pd.DataFrame:
        """获取特定时间帧的所有细胞形态"""
        
    def compute_sphericity(self, volume: float, surface: float) -> float:
        """计算球形度: (36π * V²)^(1/3) / S"""
        
    def get_features_for_cells(self,
                               cell_names: List[str],
                               time_frames: np.ndarray = None) -> pd.DataFrame:
        """批量获取细胞形态特征"""
```

---

## 5. 谱系名称匹配策略

### 5.1 命名对照

| 数据源 | 示例名称 | 格式说明 |
|-------|---------|---------|
| Large2025 | `ABplpapppa` | 完整小写路径 |
| WormGUIDES | `ABplpapppa` | 完整小写路径 |
| CShaper ContactInterface | `ABplpapppa` | 完整小写路径 |
| CShaper VolumeAndSurface | `ABplpapppa` | 完整小写路径 |
| CShaper Standard DS1 | 按树索引 | gen×pos矩阵 |

### 5.2 匹配函数

```python
def normalize_lineage_name(name: str) -> str:
    """标准化谱系名称"""
    # 移除空格、句点
    name = name.replace(" ", "").replace(".", "")
    # 标准化founder前缀大写
    for founder in ["AB", "MS", "EMS", "P0", "P1", "P2", "P3", "P4", "Z2", "Z3"]:
        if name.upper().startswith(founder):
            return founder + name[len(founder):].lower()
    # E, C, D特殊处理
    if name[0].upper() in "ECD":
        return name[0].upper() + name[1:].lower()
    return name

def lineage_to_tree_index(name: str) -> Tuple[str, int, int]:
    """
    将谱系名称转换为Standard Dataset 1的树索引
    
    Returns:
        (founder, generation, position)
        
    Example:
        'ABplpa' -> ('AB', 4, 9)  # gen=4 (从AB开始第4代), pos=9 (plpa的二进制1010=10, 0-indexed=9)
    """
```

---

## 6. 时间对齐策略

### 6.1 时间映射

```python
# CShaper: 54帧覆盖4-350细胞期
# 大约对应: 20分钟 - 380分钟 (与WormGUIDES相似)

CSHAPER_FRAMES = 54
CSHAPER_START_TIME_MIN = 20
CSHAPER_END_TIME_MIN = 380

def embryo_time_to_cshaper_frame(time_min: float) -> int:
    """将胚胎时间映射到CShaper帧"""
    if time_min < CSHAPER_START_TIME_MIN:
        return 0
    if time_min > CSHAPER_END_TIME_MIN:
        return CSHAPER_FRAMES - 1
    
    # 线性映射
    fraction = (time_min - CSHAPER_START_TIME_MIN) / (CSHAPER_END_TIME_MIN - CSHAPER_START_TIME_MIN)
    return int(fraction * (CSHAPER_FRAMES - 1))
```

### 6.2 处理时间不匹配

对于Large2025中embryo_time缺失或不在范围内的细胞:
1. 使用谱系深度估算发育阶段
2. 使用细胞类型推断时间窗口
3. 使用跨时间平均值

---

## 7. 输出规范

### 7.1 增强后的AnnData结构

```python
adata = AnnData(
    # === 原有 ===
    X=expression_matrix,                    # (n_cells, n_genes) 稀疏
    
    obs={
        # 原有
        'cell_type': ...,
        'lineage_complete': ...,
        'embryo_time_min': ...,
        'lineage_valid': ...,
        'lineage_founder': ...,
        'lineage_depth': ...,
        'has_spatial': ...,
        
        # CShaper新增
        'cell_volume': ...,                 # 细胞体积 (μm³)
        'cell_surface': ...,                # 表面积 (μm²)
        'sphericity': ...,                  # 球形度 [0,1]
        'has_morphology': ...,              # bool: 是否有形态数据
        'cshaper_frame': ...,               # 匹配的CShaper帧号
    },
    
    var={...},                              # 基因注释
    
    obsm={
        # 原有
        'X_spatial': ...,                   # (n_cells, 3) WormGUIDES坐标
        'X_lineage_binary': ...,            # (n_cells, max_depth) 谱系编码
        
        # CShaper新增
        'X_cshaper_spatial': ...,           # (n_cells, 3) CShaper标准化坐标
    },
    
    obsp={
        # 原有
        'spatial_distances': ...,           # k-NN距离
        'spatial_connectivities': ...,      # k-NN连接
        'lineage_adjacency': ...,           # 谱系邻接
        
        # CShaper新增
        'contact_adjacency': ...,           # 真实接触图 (加权)
        'contact_binary': ...,              # 真实接触图 (二值)
    },
    
    uns={
        # 原有
        'data_sources': {...},
        'build_params': {...},
        
        # CShaper新增
        'cshaper_info': {
            'samples_used': [...],
            'n_with_morphology': ...,
            'n_with_contact': ...,
            'time_mapping': {...},
        },
        'contact_statistics': {
            'mean_contacts_per_cell': ...,
            'max_contact_area': ...,
        },
    }
)
```

---

## 8. 实现优先级

### Phase 1 (MVP) - 接触图整合
- [ ] `CShaperProcessor` 基础类
- [ ] `ContactLoader` 接触矩阵加载
- [ ] `EnhancedAnnDataBuilder._add_contact_graph()`
- [ ] 测试: 接触图与k-NN图的比较

### Phase 2 - 形态特征
- [ ] `MorphologyLoader` 体积/表面积加载
- [ ] `EnhancedAnnDataBuilder._add_morphology_features()`
- [ ] 计算衍生特征 (球形度等)

### Phase 3 - 空间坐标增强
- [ ] Standard Dataset 1 HDF5解析
- [ ] 谱系→树索引转换
- [ ] `EnhancedAnnDataBuilder._enhance_spatial_coords()`

### Phase 4 - 高级功能
- [ ] 3D形态描述符 (从Standard Dataset 2)
- [ ] 时间动态接触图
- [ ] 接触图GNN集成

---

## 9. 对模型的影响

### 9.1 Spatial GNN 增强

```python
# 原来: k-NN图 (近似邻居)
edge_index = knn_graph(spatial_coords, k=10)

# 增强后: 真实接触图
edge_index = contact_adjacency.nonzero()
edge_weight = contact_adjacency.data  # 接触面积作为边权重
```

### 9.2 细胞Token特征增强

```python
# 原来: 表达 + 谱系编码
cell_features = torch.cat([expression, lineage_binary], dim=-1)

# 增强后: 表达 + 谱系编码 + 形态特征
morphology = torch.stack([volume, surface, sphericity], dim=-1)
cell_features = torch.cat([expression, lineage_binary, morphology], dim=-1)
```

### 9.3 消息传递改进

```python
# GNN消息传递时使用真实邻居关系
def forward(self, x, contact_edge_index, contact_edge_weight):
    # 邻居聚合时按接触面积加权
    return self.conv(x, contact_edge_index, edge_weight=contact_edge_weight)
```

---

## 10. 验证计划

1. **数据完整性**: 验证CShaper谱系名称与Large2025的匹配率
2. **接触图质量**: 比较接触图与k-NN图的结构差异
3. **形态特征分布**: 可视化体积/表面积随发育时间的变化
4. **下游任务**: 评估增强数据对细胞类型预测的影响
