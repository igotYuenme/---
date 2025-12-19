# 新媒体平台神秘主义内容受众画像与传播分析

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本研究通过数据挖掘和机器学习方法，对微博、B站等新媒体平台上的神秘主义内容（星座、塔罗、占卜、MBTI等）进行系统分析。研究构建了完整的数据采集、处理与分析流程，实现了受众画像聚类、传播模式分析和三维评估框架构建。

## 📋 项目概述

### 研究目标

1. **决策场景依赖分析**：分析用户在不同决策场景（情感、学业、职业）中对神秘主义内容的依赖程度
2. **受众画像聚类**：通过K-means聚类识别不同类型的用户群体及其特征
3. **内容创作者评估**：构建"内容-传播-心理"三维评估框架，评估头部内容创作者的表现

### 核心发现

- **用户类型三重分化**：心理慰藉型（25.8%）、娱乐型（47.5%）、深度参与型（26.7%）
- **场景依赖差异显著**：复合决策场景（如"日常,职业"）的依赖指数（0.5649）显著高于单一场景
- **平台内容策略差异**：微博以短文本快速传播为主，B站以深度视频互动为主

## 🗂️ 项目结构

```
news_security/
├── bilibili/                          # B站数据分析
│   ├── collect_up_videos.py          # B站视频数据采集
│   ├── q1.py                         # 决策场景分析
│   ├── longnv.py                     # 龙女塔罗UP主三维评估
│   ├── *.csv                         # 数据文件
│   └── *.png                         # 可视化图表
│
├── weibo-search-master/              # 微博数据分析
│   └── weibo/
│       ├── spiders/search.py         # 微博数据爬虫
│       ├── weibo_analyse.py          # 决策场景与依赖指数分析
│       ├── user_portrait_analysis.py # 用户画像聚类分析
│       ├── taobaibai.py              # 陶白白博主三维评估
│       ├── collect_taobaibai_weibo.py # 博主数据采集
│       ├── *.json                    # 数据文件
│       └── *.png                     # 可视化图表
│
├── xiaohongshu/                      # 小红书数据分析（可选）
│   ├── xiaohongshu_data.py
│   └── user_portrait_analysis.py
│
├── 报告/                             # 实验报告
│   └── 新闻安全.tex                  # LaTeX报告源文件
│
└── requirements.txt                  # Python依赖包
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- LaTeX（用于编译报告，可选）

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `pandas` - 数据处理
- `numpy` - 数值计算
- `scikit-learn` - 机器学习（K-means聚类）
- `matplotlib` / `seaborn` - 数据可视化
- `jieba` - 中文分词
- `scrapy` - 网络爬虫框架

### 数据采集

#### 1. 微博数据采集

```bash
cd weibo-search-master/weibo
scrapy crawl weibo_search
```

#### 2. B站数据采集

```bash
cd bilibili
python collect_up_videos.py
```

#### 3. 特定博主数据采集

```bash
cd weibo-search-master/weibo
python collect_taobaibai_weibo.py
```

### 数据分析

#### 1. 微博决策场景分析

```bash
cd weibo-search-master/weibo
python weibo_analyse.py
```

输出：
- `weibo_scenario_distribution.png` - 场景分布饼图
- `weibo_scenario_sentiment.png` - 场景情感分析
- `weibo_scenario_dependence.png` - 场景依赖指数图
- `weibo_user_clustering.png` - 用户聚类散点图

#### 2. B站决策场景分析

```bash
cd bilibili
python q1.py
```

输出：
- `scene_ratio_bar_en.png` - 场景占比柱状图
- `scene_ratio_pie_en.png` - 场景分布饼图
- `other_subscene_bar_en.png` - Other类别细分柱状图

#### 3. 用户画像聚类分析

```bash
cd weibo-search-master/weibo
python user_portrait_analysis.py
```

输出：
- `user_portrait_radar.png` - 用户特征雷达图
- `user_portrait_heatmap.png` - 特征相关性热力图
- `user_portrait_stacked_bar.png` - 内容类型堆叠分布
- `weibo_portrait_report.txt` - 用户画像分析报告

#### 4. 博主三维评估

**陶白白评估：**
```bash
cd weibo-search-master/weibo
python taobaibai.py
```

**龙女塔罗评估：**
```bash
cd bilibili
python longnv.py
```

输出：
- `blogger_enhanced_assessment.png` / `longnv_enhanced_assessment.png` - 三维评估可视化
- `blogger_enhanced_results_*.json` - 评估结果JSON文件

## 📊 核心方法

### 1. 决策场景标注

基于关键词匹配将内容分类为：
- **情感场景**（Emotion）：复合、分手、恋爱等
- **学业场景**（Study）：考试、考研、毕业等
- **职业场景**（Career）：面试、求职、工作等
- **日常场景**（Daily）：运势、水逆等

### 2. 神秘依赖指数（MDI）构建

$$
\text{MDI} = 0.4 \times Z(\text{神秘词汇密度}) + 0.4 \times Z(\text{互动强度}) - 0.2 \times Z(\text{情感得分})
$$

其中：
- **神秘词汇密度**：统计关键词（星座、塔罗、占卜等）出现频次
- **互动强度**：转发数 + 评论数×2 + 点赞数×0.5
- **情感得分**：基于情感分析，负面情绪增强依赖

### 3. 用户聚类

使用K-means算法，基于以下特征对用户进行聚类：
- 内容偏好（学业/职业、情感、娱乐）
- 互动强度（对数变换）
- 时间特征（考试周、招聘季、休闲时段）
- 心理慰藉需求

### 4. 三维评估框架

构建"内容-传播-心理"三维评估体系：

| 维度 | 评估指标 | 权重 |
|------|---------|------|
| **内容维度** | 文本长度、主题多样性、理性分析比例 | 40% |
| **传播维度** | 话题覆盖率、参与用户数、平均互动数 | 35% |
| **心理维度** | 情绪分布、心理需求、支持指数 | 25% |

## 📈 主要结果

### 用户类型分布

- **心理慰藉型（25.8%）**：考试周发帖比例93.2%，主要关注学业和职业相关话题
- **娱乐型（47.5%）**：规模最大，以社交娱乐为主要目的
- **深度参与型（26.7%）**：招聘季活跃度92.9%，具有高参与度和付费意愿

### 场景依赖指数

- **最高**："日常,职业"组合场景（0.5649）
- **次之**："职业,情感"场景（0.4653）
- **最低**：单一"职业"场景（-0.2422）、"学业"场景（-0.1870）

### 博主评估结果

**陶白白Sensei**：
- 综合评分：43.6分
- 内容维度：30.7分（情感咨询42.8%，星座运势31.1%）
- 传播维度：70.1分（话题覆盖率100%）
- 心理维度：33.8分（情感需求55.6%）

**龙女塔罗**：
- 综合评分：32.5分
- 内容维度：29.4分（塔罗占卜96.2%，情感咨询30.3%）
- 传播维度：50.8分（平均播放量634,613次）
- 心理维度：11.7分（心理支持功能较弱）

## 📄 报告生成

使用LaTeX编译实验报告：

```bash
cd 报告
pdflatex 新闻安全.tex
```

报告包含：
- 数据采集方法
- 数据分析流程
- 可视化结果
- 研究发现总结
- 讨论与展望

## 🔧 配置说明

### 微博爬虫配置

在 `weibo-search-master/weibo/settings.py` 中配置：
- Cookie信息
- 请求头
- 爬取延迟

### 数据文件路径

各脚本中的默认数据文件路径：
- 微博数据：`weibo-search-master/weibo/weibo_data_*.json`
- B站数据：`bilibili/bilibili_videos.csv`
- 用户画像数据：`weibo-search-master/weibo/weibo_user_portrait.csv`

## 📝 注意事项

1. **数据采集**：需要配置相应的Cookie和请求头，遵守平台爬虫协议
2. **数据隐私**：所有用户数据已匿名化处理，仅用于学术研究
3. **依赖版本**：建议使用虚拟环境管理依赖包版本

## 📚 引用

如果使用本项目，请引用：

```bibtex
@misc{news_security_2024,
  title={新媒体平台神秘主义内容受众画像与传播分析},
  author={吴羽宸},
  year={2024},
  institution={武汉大学国家网络安全学院}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👤 作者

- **吴羽宸** - 网络空间安全2023级3班
- 学号：2023302181006
- 单位：武汉大学国家网络安全学院

## 🙏 致谢

- 感谢所有提供数据支持的平台
- 感谢开源社区提供的优秀工具和库
<<<<<<< HEAD
=======

>>>>>>> 63cc53d (本地 README 修改)
