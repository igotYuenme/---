# weibo_analyse.py 生成的图片分析

## 概述

`weibo_analyse.py` 脚本共生成 **4张图片**，用于分析微博用户在决策场景中对玄学内容的提及和依赖情况。

---

## 一、图片清单

### 1. weibo_scenario_distribution.png ⭐
**决策场景分布饼图**

- **生成代码位置**：第156-168行
- **图表类型**：饼图（Pie Chart）
- **图表大小**：10x10 英寸
- **DPI**：300
- **标题**："Distribution of Decision Scenarios in Mystic-related Posts"

**分析内容**：
- 展示各决策场景（情感/学业/职业/日常/Other）在玄学相关内容中的分布占比
- 百分比显示（autopct='%1.1f%%'）
- 起始角度90度

**研究价值**：
- 识别用户提及玄学内容的主要场景分布
- 了解哪个场景下用户最常提及玄学内容
- 如果"Other"比例过高（>20%），说明关键词列表需要扩展

**数据来源**：
```python
scene_dist = df['scene_tag'].value_counts()
```

---

### 2. weibo_scenario_sentiment.png
**各场景情感分析柱状图**

- **生成代码位置**：第208-218行
- **图表类型**：水平柱状图（Horizontal Bar Chart）
- **图表大小**：8x5 英寸
- **DPI**：300
- **标题**："Average Sentiment Score by Decision Scenario"

**分析内容**：
- 计算每个决策场景下的平均情感得分
- 情感得分计算公式：
  ```python
  sentiment_score = positive_count - negative_count
  ```
- 正面词汇：'顺利', '开心', '希望', '成功', '上岸', '幸运', '期待'
- 负面词汇：'焦虑', '难受', '崩溃', '害怕', '迷茫', '失败', '压力', 'emo'
- 灰色虚线标记0分线（区分正面/负面情绪）

**研究价值**：
- 了解不同场景下用户的情感状态
- 识别哪些场景下用户更焦虑（负分）或更积极（正分）
- 情感得分越高，说明该场景下用户的情绪越积极

**数据来源**：
```python
scene_sentiment = df.groupby('scene_tag')['sentiment_score'].mean()
```

---

### 3. weibo_scenario_dependence.png ⭐⭐⭐ **核心图表**
**各场景神秘依赖指数柱状图**

- **生成代码位置**：第247-257行
- **图表类型**：柱状图（Bar Chart）
- **图表大小**：8x5 英寸
- **DPI**：300
- **标题**："Mystic Dependence Index by Decision Scenario"
- **排序方式**：按依赖指数降序排列（ascending=False）

**分析内容**：
- **神秘依赖指数**（Dependence Index）的计算：
  1. **神秘词汇密度**（mystic_density）：统计文本中包含的神秘词汇数量
     - 神秘词汇：'星座', '塔罗', '占卜', '显化', '运势', '宇宙', '水逆', '玄学'
  2. **互动分数**（interaction_score）：
     ```python
     interaction_score = reposts_count + comments_count * 2 + attitudes_count * 0.5
     ```
  3. **情感得分**（sentiment_score）：见上文
  4. **标准化**：对以上三个特征进行标准化（StandardScaler）
  5. **综合指数**：
     ```python
     depend_index = scaled_mystic_density * 0.4 + 
                    scaled_interaction_score * 0.4 + 
                    (-scaled_sentiment_score) * 0.2
     ```
     - 权重：神秘词汇密度40%，互动强度40%，负面情绪20%

**研究价值**：⭐⭐⭐ **核心图表**
- **直接回答核心研究问题**："用户在什么场景下更依赖玄学内容"
- 依赖指数越高，说明该场景下用户：
  - 更频繁地提及神秘词汇
  - 互动更积极
  - 情绪更焦虑（负面情绪越多，依赖指数越高）
- 用于识别用户在哪些决策场景下对玄学内容的需求最强烈

**数据来源**：
```python
scene_depend = df.groupby('scene_tag')['depend_index'].mean()
```

**建议使用场景**：
- 论文/报告中的核心发现图表
- 用于说明不同场景下用户对玄学内容的依赖程度差异

---

### 4. weibo_user_clustering.png
**用户聚类散点图**

- **生成代码位置**：第270-286行
- **图表类型**：散点图（Scatter Plot）
- **图表大小**：9x6 英寸
- **DPI**：300
- **标题**："User Clustering Based on Mystic Engagement"
- **颜色映射**：viridis 色彩映射（根据cluster ID）

**分析内容**：
- **聚类方法**：K-means聚类（n_clusters=3）
- **聚类特征**：
  1. **对数互动强度**（log_interaction）：
     ```python
     log_interaction = log10(interaction_score + 1)
     ```
  2. **神秘依赖指数**（depend_index）：见上文
- **聚类数量**：3个簇（cluster）
- **标准化**：对特征进行标准化处理

**研究价值**：
- 识别不同类型的用户群体
- 分析用户的神秘依赖程度与互动强度的关系
- 发现高依赖低互动、低依赖高互动等不同的用户模式
- 为后续用户画像分析提供基础

**数据来源**：
```python
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

**可视化特点**：
- X轴：对数互动强度（Log Interaction Intensity）
- Y轴：神秘依赖指数（Mystic Dependence Index）
- 颜色条：显示不同的聚类ID
- 透明度：0.6（便于观察重叠的点）

---

## 二、图片之间的关系

### 分析流程：
1. **场景分布**（weibo_scenario_distribution.png）
   ↓ 了解各场景的基本分布
2. **场景情感**（weibo_scenario_sentiment.png）
   ↓ 了解各场景下的情绪状态
3. **场景依赖指数**（weibo_scenario_dependence.png）⭐⭐⭐
   ↓ **核心发现：哪个场景下用户最依赖玄学**
4. **用户聚类**（weibo_user_clustering.png）
   ↓ 识别不同用户群体的特征

### 核心研究问题：
**"用户在什么场景下更主动提及/依赖玄学内容？"**

**答案主要来自**：`weibo_scenario_dependence.png`
- 依赖指数最高的场景，说明用户在该场景下最依赖玄学内容
- 可以结合 `weibo_scenario_sentiment.png` 解释为什么依赖（情绪焦虑程度）

---

## 三、图片使用建议

### 论文/报告中的使用顺序：

1. **核心图表**（必用）：
   - `weibo_scenario_dependence.png` ⭐⭐⭐
     - 作为主要发现图表
     - 重点解释依赖指数的计算方法和意义
     - 说明哪些场景下用户最依赖玄学内容

2. **支撑图表**（推荐使用）：
   - `weibo_scenario_distribution.png`
     - 说明各场景的基本分布情况
     - 提供背景信息
   - `weibo_scenario_sentiment.png`
     - 解释依赖指数的情绪维度
     - 说明为什么某些场景下依赖程度高（焦虑情绪）

3. **补充图表**（可选）：
   - `weibo_user_clustering.png`
     - 如果报告包含用户群体分析部分
     - 说明用户群体的多样性

---

## 四、技术细节

### 数据预处理：
- 文本清洗：去除HTML标签、URL、@提及、#话题
- 列名标准化：统一 `reposts_count`, `comments_count`, `attitudes_count`

### 场景标注逻辑：
- 关键词匹配（优先匹配）
- 多场景匹配（返回匹配度最高的2个场景）
- 单字匹配（fallback）

### 数据质量检查：
- 检查"Other"比例（如果>20%会警告）
- 建议扩展关键词列表

---

## 五、注意事项

1. **图片格式**：
   - 所有图片均为PNG格式
   - DPI=300，适合论文/报告使用
   - 使用 `bbox_inches='tight'` 确保完整显示

2. **字体设置**：
   - 使用Arial字体（适合英文标题）
   - 图表标题为英文，便于国际化

3. **数据依赖**：
   - 需要 `weibo_data_20251218_163102.json` 数据文件
   - 确保数据中包含必要的列：`text`, `reposts_count`, `comments_count`, `attitudes_count`

4. **与B站分析的对比**：
   - 微博场景分析：`weibo_scenario_dependence.png`
   - B站场景分析：`scene_ratio_bar_en.png`
   - 两个图表可以对比，展示不同平台用户在场景依赖上的差异

---

## 六、文件路径

所有图片保存在：
```
D:\news_security\weibo-search-master\weibo\
```

1. `weibo_scenario_distribution.png`
2. `weibo_scenario_sentiment.png`
3. `weibo_scenario_dependence.png` ⭐⭐⭐
4. `weibo_user_clustering.png`

