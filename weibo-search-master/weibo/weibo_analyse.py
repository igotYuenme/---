import json
import re
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ===== matplotlib 英文字体设置 =====
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================
# 1. Load data
# =====================================================
DATA_FILE = 'weibo_data_20251218_163102.json'

print(f"📥 正在加载数据文件: {DATA_FILE}")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"✅ 成功加载 {len(df)} 条微博数据")

# 确保列名一致
if 'reposts_count' not in df.columns:
    if 'reposts' in df.columns:
        df['reposts_count'] = df['reposts']
    else:
        df['reposts_count'] = 0
    
if 'comments_count' not in df.columns:
    if 'comments' in df.columns:
        df['comments_count'] = df['comments']
    else:
        df['comments_count'] = 0
    
if 'attitudes_count' not in df.columns:
    if 'likes' in df.columns:
        df['attitudes_count'] = df['likes']
    else:
        df['attitudes_count'] = 0

print(f"📊 数据列: {df.columns.tolist()}")
print(f"📊 数据预览:\n{df.head()}")

# =====================================================
# 2. Text cleaning
# =====================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@.*?\s', '', text)
    text = re.sub(r'#.*?#', '', text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# =====================================================
# 2.1 Decision scene tagging (优化版 - 扩展关键词以降低Other比例)
# =====================================================
decision_scenes = {
    "Emotion": [
        "复合", "分手", "恋爱", "喜欢", "前任", "暧昧", "桃花", "婚姻", 
        "感情", "情感", "爱情", "对象", "恋人", "情侣", "相亲", "脱单",
        "追求", "表白", "暗恋", "单恋", "失恋", "挽回", "和好", "冷战",
        "吵架", "矛盾", "异地", "异地恋", "结婚", "离婚", "单身"
    ],
    "Study": [
        "考试", "考研", "毕业", "论文", "复习", "四六级", "教资", "专四",
        "学习", "备考", "上岸", "录取", "成绩", "挂科", "补考", "重修",
        "期末", "期中", "作业", "课程", "专业", "选课", "选专业", "转专业",
        "保研", "出国", "留学", "语言", "英语", "托福", "雅思", "GRE",
        "学历", "学位", "证书", "资格证", "教师", "公务员", "事业单位"
    ],
    "Career": [
        "工作", "面试", "求职", "offer", "跳槽", "事业", "岗位",
        "职业", "就业", "应聘", "简历", "HR", "薪资", "工资", "薪水",
        "转正", "实习", "试用期", "离职", "辞职", "被辞", "裁员",
        "升职", "加薪", "同事", "领导", "老板", "团队", "项目",
        "创业", "公司", "企业", "行业", "职位", "招聘", "投递"
    ],
    "Daily": [
        "水逆", "运势", "选择", "健康", "出行", "今天", "本周",
        "星座", "占卜", "塔罗", "星盘", "占星", "命理", "玄学",
        "预测", "分析", "建议", "指导", "咨询", "答疑", "解惑",
        "迷茫", "困惑", "焦虑", "压力", "烦恼", "纠结", "犹豫",
        "决定", "决策", "选择困难", "人生", "未来", "规划", "目标",
        "生活", "日常", "习惯", "改变", "改善", "提升", "成长"
    ]
}

def tag_scene(text):
    """场景标记函数 - 改进版：允许多标签，降低Other比例"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 'Other'
    
    hits = []
    # 统计每个场景的匹配数量
    scene_scores = {}
    
    for scene, words in decision_scenes.items():
        count = sum(1 for w in words if w in text)
        if count > 0:
            scene_scores[scene] = count
            hits.append(scene)
    
    # 如果有匹配，返回匹配的场景（按匹配度排序）
    if hits:
        # 按匹配数量排序，返回前2个最重要的场景
        sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
        return ','.join([s[0] for s in sorted_scenes[:2]])
    
    # 如果没有任何匹配，尝试更宽松的匹配（检查关键词的子串）
    # 这是最后的手段，尽量降低Other的比例
    loose_keywords = {
        "Emotion": ["爱", "情", "恋", "婚"],
        "Study": ["学", "考", "试", "书"],
        "Career": ["工", "职", "业", "作"],
        "Daily": ["运", "势", "星", "占", "问", "题", "想", "要"]
    }
    
    for scene, loose_words in loose_keywords.items():
        if any(w in text for w in loose_words):
            return scene  # 使用宽松匹配，至少给一个分类
    
    return 'Other'

df['scene_tag'] = df['clean_text'].apply(tag_scene)

# 统计场景分布（处理多标签情况）
scene_dist = df['scene_tag'].str.split(',').explode().value_counts()

# 打印统计信息
print(f"\n📊 场景分布统计:")
print(f"   总样本数: {len(df)}")
print(f"   场景标签分布:")
print("   " + "="*50)
scene_dist_dict = {}
for scene, count in scene_dist.items():
    ratio = count / len(df) * 100
    scene_dist_dict[scene] = {'count': count, 'ratio': ratio}
    print(f"     {scene:15s}: {count:4d} 条 ({ratio:5.2f}%)")
print("   " + "="*50)

# 计算Other的比例
other_ratio = scene_dist.get('Other', 0) / len(df) * 100
if other_ratio > 20:
    print(f"\n⚠️ 注意: Other比例较高 ({other_ratio:.1f}%)")
    print(f"   建议: 可以进一步扩展关键词列表或调整匹配逻辑")
else:
    print(f"\n✅ Other比例: {other_ratio:.1f}% (已优化)")

# ---- Pie chart ----
plt.figure(figsize=(10, 10))
scene_dist.plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 10}
)
plt.title("Distribution of Decision Scenarios in Mystic-related Posts", fontsize=14, fontweight='bold')
plt.ylabel("")
plt.tight_layout()
plt.savefig('weibo_scenario_distribution.png', dpi=300, bbox_inches='tight')
print("\n💾 已保存决策场景分布图: weibo_scenario_distribution.png")
print("   📊 图片中的具体数值:")
for scene in scene_dist.index:
    count = scene_dist[scene]
    ratio = count / len(df) * 100
    print(f"      {scene}: {count}条 ({ratio:.1f}%)")
plt.show()

# =====================================================
# 2.3 Topic Modeling (LDA)
# =====================================================
def cut_words(text):
    return ' '.join(jieba.cut(text))

df['cut_text'] = df['clean_text'].apply(cut_words)

vectorizer = CountVectorizer(min_df=5, max_df=0.8)
dtm = vectorizer.fit_transform(df['cut_text'])

lda = LatentDirichletAllocation(
    n_components=4,
    random_state=42
)
lda.fit(dtm)

print("\n📊 LDA主题建模结果:")
print("   " + "="*60)
for idx, topic in enumerate(lda.components_):
    words = [vectorizer.get_feature_names_out()[i]
             for i in topic.argsort()[:-11:-1]]
    print(f"   Topic {idx}: {' '.join(words)}")
print("   " + "="*60)

# =====================================================
# 2.4 Sentiment analysis
# =====================================================
positive_words = ['顺利', '开心', '希望', '成功', '上岸', '幸运', '期待']
negative_words = ['焦虑', '难受', '崩溃', '害怕', '迷茫', '失败', '压力', 'emo']

def sentiment_score(text):
    pos = sum(1 for w in positive_words if w in text)
    neg = sum(1 for w in negative_words if w in text)
    return pos - neg

df['sentiment_score'] = df['clean_text'].apply(sentiment_score)

scene_sentiment = df.groupby('scene_tag')['sentiment_score'].mean()

# 打印情感得分统计
print(f"\n📊 各场景情感得分统计:")
print("   " + "="*60)
print(f"   {'场景':15s} {'平均情感得分':12s} {'解释':30s}")
print("   " + "-"*60)
scene_sentiment_sorted = scene_sentiment.sort_values()
for scene, score in scene_sentiment_sorted.items():
    interpretation = "正面情绪" if score > 0 else "负面情绪（焦虑）" if score < 0 else "中性"
    print(f"   {scene:15s} {score:12.3f} {interpretation:30s}")
print("   " + "="*60)
print("   说明: 正值=正面情绪，负值=负面情绪/焦虑，0=中性")

# ---- Bar chart (sentiment) ----
plt.figure(figsize=(8, 5))
scene_sentiment.sort_values().plot(kind='barh')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Average Sentiment Score by Decision Scenario")
plt.xlabel("Sentiment Score (Negative = Anxiety)")
plt.ylabel("Decision Scenario")
plt.tight_layout()
plt.savefig('weibo_scenario_sentiment.png', dpi=300, bbox_inches='tight')
print("\n💾 已保存各场景情感分析图: weibo_scenario_sentiment.png")
print("   📊 图片中的具体数值:")
for scene, score in scene_sentiment_sorted.items():
    print(f"      {scene}: {score:.3f}")
plt.show()

# =====================================================
# 2.5 Mystic Dependence Index
# =====================================================
mystic_words = ['星座', '塔罗', '占卜', '显化', '运势', '宇宙', '水逆', '玄学']

def mystic_density(text):
    return sum(1 for w in mystic_words if w in text)

df['mystic_density'] = df['clean_text'].apply(mystic_density)

df['interaction_score'] = (
    df['reposts_count'] +
    df['comments_count'] * 2 +
    df['attitudes_count'] * 0.5
)

depend_features = df[['mystic_density', 'interaction_score', 'sentiment_score']].fillna(0)
depend_scaled = StandardScaler().fit_transform(depend_features)

df['depend_index'] = (
    depend_scaled[:, 0] * 0.4 +
    depend_scaled[:, 1] * 0.4 +
    (-depend_scaled[:, 2]) * 0.2
)

scene_depend = df.groupby('scene_tag')['depend_index'].mean()

# 打印依赖指数统计（按降序排列）
scene_depend_sorted = scene_depend.sort_values(ascending=False)
print(f"\n📊 各场景神秘依赖指数统计（核心指标）:")
print("   " + "="*70)
print(f"   {'场景':15s} {'依赖指数':12s} {'排名':6s} {'解释':30s}")
print("   " + "-"*70)
max_depend = scene_depend_sorted.max()
min_depend = scene_depend_sorted.min()
for rank, (scene, depend_idx) in enumerate(scene_depend_sorted.items(), 1):
    if depend_idx == max_depend:
        interpretation = "依赖程度最高 ⭐"
    elif depend_idx == min_depend:
        interpretation = "依赖程度最低"
    elif depend_idx > 0:
        interpretation = "依赖程度较高"
    else:
        interpretation = "依赖程度较低"
    print(f"   {scene:15s} {depend_idx:12.4f} #{rank:<5d} {interpretation:30s}")
print("   " + "="*70)
print(f"   依赖指数范围: {min_depend:.4f} ~ {max_depend:.4f}")
print(f"   指数最高场景: {scene_depend_sorted.index[0]} ({max_depend:.4f})")
print("   说明: 依赖指数越高，说明该场景下用户对玄学内容的需求越强烈")
print("         指数计算基于: 神秘词汇密度(40%) + 互动强度(40%) + 负面情绪(20%)")

# ---- Bar chart (dependence) ----
plt.figure(figsize=(8, 5))
scene_depend.sort_values(ascending=False).plot(kind='bar')
plt.title("Mystic Dependence Index by Decision Scenario")
plt.ylabel("Dependence Index")
plt.xlabel("Decision Scenario")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('weibo_scenario_dependence.png', dpi=300, bbox_inches='tight')
print("\n💾 已保存各场景神秘依赖指数图: weibo_scenario_dependence.png")
print("   📊 图片中的具体数值（按依赖指数降序）:")
for scene, depend_idx in scene_depend_sorted.items():
    print(f"      {scene}: {depend_idx:.4f}")
plt.show()

# =====================================================
# 3. User clustering
# =====================================================
df['log_interaction'] = np.log10(df['interaction_score'] + 1)

X = df[['log_interaction', 'depend_index']].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 打印聚类结果统计
print(f"\n📊 用户聚类结果统计:")
print("   " + "="*70)
cluster_stats = df.groupby('cluster').agg({
    'log_interaction': ['count', 'mean', 'std'],
    'depend_index': ['mean', 'std'],
    'interaction_score': 'mean',
    'mystic_density': 'mean',
    'sentiment_score': 'mean'
}).round(4)

print(f"   {'聚类ID':8s} {'样本数':8s} {'对数互动(均值)':15s} {'依赖指数(均值)':15s} {'互动分数(均值)':15s}")
print("   " + "-"*70)
for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    count = len(cluster_data)
    log_int_mean = cluster_data['log_interaction'].mean()
    depend_mean = cluster_data['depend_index'].mean()
    int_score_mean = cluster_data['interaction_score'].mean()
    mystic_mean = cluster_data['mystic_density'].mean()
    sent_mean = cluster_data['sentiment_score'].mean()
    ratio = count / len(df) * 100
    
    print(f"   Cluster {cluster_id:<4d} {count:8d} {log_int_mean:15.4f} {depend_mean:15.4f} {int_score_mean:15.2f}")
    print(f"             ({ratio:5.1f}%)  神秘词汇密度: {mystic_mean:.3f}, 情感得分: {sent_mean:.3f}")

print("   " + "="*70)
print(f"   总样本数: {len(df)}")
print(f"   聚类特征: 对数互动强度 vs 神秘依赖指数")

# ---- Scatter plot ----
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    df['log_interaction'],
    df['depend_index'],
    c=df['cluster'],
    cmap='viridis',
    alpha=0.6
)
plt.xlabel("Log Interaction Intensity")
plt.ylabel("Mystic Dependence Index")
plt.title("User Clustering Based on Mystic Engagement")
plt.colorbar(scatter, label="Cluster ID")
plt.tight_layout()
plt.savefig('weibo_user_clustering.png', dpi=300, bbox_inches='tight')
print("\n💾 已保存用户聚类散点图: weibo_user_clustering.png")
print("   📊 图片中的聚类结果:")
for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    count = len(cluster_data)
    ratio = count / len(df) * 100
    print(f"      Cluster {cluster_id}: {count}个用户 ({ratio:.1f}%)")
    print(f"         - 对数互动强度范围: {cluster_data['log_interaction'].min():.3f} ~ {cluster_data['log_interaction'].max():.3f} (均值: {cluster_data['log_interaction'].mean():.3f})")
    print(f"         - 依赖指数范围: {cluster_data['depend_index'].min():.4f} ~ {cluster_data['depend_index'].max():.4f} (均值: {cluster_data['depend_index'].mean():.4f})")
plt.show()
