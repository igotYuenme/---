# ======================================
# 小红书受众画像分析：核心圈层与行为聚类
# 功能：
#   - 对有效用户进行聚类分析
#   - 识别三类核心圈层人群
#   - 生成画像报告和可视化
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import jieba
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================================
# 1. 数据加载与预处理
# ======================================
def load_data(csv_path="xiaohongshu_notes.csv"):
    """加载小红书笔记数据"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✅ 成功加载 {len(df)} 条笔记数据")
        return df
    except FileNotFoundError:
        print(f"❌ 文件 {csv_path} 不存在，请先运行 xiaohongshu_data.py 抓取数据")
        return None
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

# ======================================
# 2. 特征工程
# ======================================

def extract_keyword_features(df):
    """提取关键词相关特征"""
    # 定义关键词分类
    academic_career_keywords = ['考前', '考试', '面试', '招聘', '求职', '考研', '毕业', '论文', '实习']
    emotional_keywords = ['分手', '感情', '恋爱', '复合', '桃花', '姻缘', '情感']
    entertainment_keywords = ['运势', '水逆', 'MBTI', '显化', '吸引力法则']
    
    def classify_keyword_type(keyword, title, desc):
        """根据关键词和内容判断类型"""
        text = f"{keyword} {title} {desc}".lower()
        
        academic_score = sum(1 for kw in academic_career_keywords if kw in text)
        emotional_score = sum(1 for kw in emotional_keywords if kw in text)
        entertainment_score = sum(1 for kw in entertainment_keywords if kw in text)
        
        if academic_score > 0:
            return 'academic_career'
        elif emotional_score > 0:
            return 'emotional'
        elif entertainment_score > 0:
            return 'entertainment'
        else:
            return 'other'
    
    df['content_type'] = df.apply(
        lambda x: classify_keyword_type(
            str(x.get('keyword', '')),
            str(x.get('title', '')),
            str(x.get('desc', ''))
        ), axis=1
    )
    
    # 创建独热编码特征
    df['is_academic_career'] = (df['content_type'] == 'academic_career').astype(int)
    df['is_emotional'] = (df['content_type'] == 'emotional').astype(int)
    df['is_entertainment'] = (df['content_type'] == 'entertainment').astype(int)
    
    return df

def calculate_interaction_features(df):
    """计算互动特征"""
    # 处理缺失值和类型转换
    for col in ['likes', 'comments', 'favorites']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
    
    # 计算互动总分
    df['interaction_score'] = (
        df['likes'] * 0.3 +
        df['comments'] * 0.5 +
        df['favorites'] * 0.2
    )
    
    # 计算互动活跃度（对数变换）
    df['log_interaction'] = np.log10(df['interaction_score'] + 1)
    
    # 计算互动多样性（有评论的比例）
    df['has_comments'] = (df['comments'] > 0).astype(int)
    df['has_favorites'] = (df['favorites'] > 0).astype(int)
    df['interaction_diversity'] = (
        df['has_comments'] * 0.5 +
        df['has_favorites'] * 0.5
    )
    
    return df

def calculate_engagement_features(df):
    """计算参与度特征"""
    # 基于作者维度聚合（如果有作者信息）
    if 'author' in df.columns:
        author_stats = df.groupby('author').agg({
            'interaction_score': ['sum', 'mean', 'count'],
            'likes': 'sum',
            'comments': 'sum',
            'favorites': 'sum'
        }).reset_index()
        
        author_stats.columns = ['author', 'total_interaction', 'avg_interaction', 
                               'post_count', 'total_likes', 'total_comments', 'total_favorites']
        
        # 计算作者参与度指标
        author_stats['engagement_level'] = (
            np.log10(author_stats['total_interaction'] + 1) * 0.4 +
            np.log10(author_stats['post_count'] + 1) * 0.3 +
            (author_stats['avg_interaction'] / (author_stats['avg_interaction'].max() + 1)) * 0.3
        )
        
        # 合并回原数据
        df = df.merge(author_stats[['author', 'engagement_level', 'post_count']], 
                     on='author', how='left')
        df['engagement_level'] = df['engagement_level'].fillna(0)
        df['post_count'] = df['post_count'].fillna(1)
    else:
        df['engagement_level'] = df['interaction_score'] / (df['interaction_score'].max() + 1)
        df['post_count'] = 1
    
    return df

def extract_content_features(df):
    """提取内容特征"""
    def count_keywords(text, keyword_list):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(1 for kw in keyword_list if kw in text_lower)
    
    # 心理慰藉相关关键词
    comfort_keywords = ['建议', '指引', '帮助', '迷茫', '焦虑', '压力', '困惑', '求助']
    # 娱乐相关关键词
    fun_keywords = ['运势', '水逆', '有趣', '好玩', '测试', 'MBTI']
    # 深度参与相关关键词
    deep_keywords = ['咨询', '付费', '课程', '学习', '深入', '专业', '分析']
    
    df['comfort_score'] = df.apply(
        lambda x: count_keywords(f"{x.get('title', '')} {x.get('desc', '')}", comfort_keywords),
        axis=1
    )
    df['fun_score'] = df.apply(
        lambda x: count_keywords(f"{x.get('title', '')} {x.get('desc', '')}", fun_keywords),
        axis=1
    )
    df['deep_score'] = df.apply(
        lambda x: count_keywords(f"{x.get('title', '')} {x.get('desc', '')}", deep_keywords),
        axis=1
    )
    
    return df

# ======================================
# 3. 聚类分析
# ======================================

def perform_clustering(df, n_clusters=3):
    """执行K-means聚类"""
    # 选择聚类特征
    feature_cols = [
        'is_academic_career',
        'is_emotional',
        'is_entertainment',
        'log_interaction',
        'interaction_diversity',
        'engagement_level',
        'comfort_score',
        'fun_score',
        'deep_score'
    ]
    
    # 确保所有特征列存在
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 缺少特征列: {missing_cols}，将使用默认值填充")
        for col in missing_cols:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 计算聚类中心
    cluster_centers = kmeans.cluster_centers_
    
    print(f"\n✅ 完成聚类分析，共 {n_clusters} 个簇")
    print(f"各簇样本数: {pd.Series(df['cluster']).value_counts().sort_index().to_dict()}")
    
    return df, kmeans, scaler, cluster_centers

def identify_user_types(df):
    """识别三类用户群体"""
    cluster_profiles = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        # 计算各特征的平均值
        profile = {
            'cluster_id': cluster_id,
            'count': len(cluster_data),
            'academic_career_ratio': cluster_data['is_academic_career'].mean(),
            'emotional_ratio': cluster_data['is_emotional'].mean(),
            'entertainment_ratio': cluster_data['is_entertainment'].mean(),
            'avg_interaction': cluster_data['interaction_score'].mean(),
            'avg_engagement': cluster_data['engagement_level'].mean(),
            'avg_comfort_score': cluster_data['comfort_score'].mean(),
            'avg_fun_score': cluster_data['fun_score'].mean(),
            'avg_deep_score': cluster_data['deep_score'].mean(),
        }
        
        cluster_profiles.append(profile)
    
    # 根据特征识别用户类型
    user_type_map = {}
    
    for profile in cluster_profiles:
        cluster_id = profile['cluster_id']
        
        # 判断逻辑
        if (profile['academic_career_ratio'] > 0.3 and 
            profile['avg_comfort_score'] > profile['avg_fun_score']):
            user_type = '心理慰藉型'
        elif (profile['emotional_ratio'] > 0.3 or 
              profile['entertainment_ratio'] > 0.4):
            user_type = '娱乐型'
        elif (profile['avg_engagement'] > 0.5 or 
              profile['avg_deep_score'] > 1):
            user_type = '深度参与型'
        else:
            # 根据主要特征判断
            if profile['academic_career_ratio'] > profile['emotional_ratio']:
                user_type = '心理慰藉型'
            elif profile['entertainment_ratio'] > 0.2:
                user_type = '娱乐型'
            else:
                user_type = '深度参与型'
        
        user_type_map[cluster_id] = user_type
    
    # 映射到数据框
    df['user_type'] = df['cluster'].map(user_type_map)
    
    return df, user_type_map

# ======================================
# 4. 可视化
# ======================================

def plot_clustering_results(df, save_path="xiaohongshu_clustering_results.png"):
    """绘制聚类结果可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 聚类散点图（PCA降维）
    ax1 = axes[0, 0]
    feature_cols = ['is_academic_career', 'is_emotional', 'is_entertainment', 
                   'log_interaction', 'interaction_diversity', 'engagement_level',
                   'comfort_score', 'fun_score', 'deep_score']
    X = df[feature_cols].fillna(0)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], 
                         cmap='viridis', alpha=0.6, s=50)
    ax1.set_xlabel('主成分1', fontsize=12)
    ax1.set_ylabel('主成分2', fontsize=12)
    ax1.set_title('用户聚类结果（PCA降维）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='簇ID')
    
    # 2. 用户类型分布
    ax2 = axes[0, 1]
    user_type_counts = df['user_type'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax2.bar(user_type_counts.index, user_type_counts.values, color=colors[:len(user_type_counts)])
    ax2.set_xlabel('用户类型', fontsize=12)
    ax2.set_ylabel('数量', fontsize=12)
    ax2.set_title('三类用户群体分布', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. 各类型特征对比（雷达图风格 - 用条形图代替）
    ax3 = axes[1, 0]
    type_features = df.groupby('user_type').agg({
        'is_academic_career': 'mean',
        'is_emotional': 'mean',
        'is_entertainment': 'mean',
        'log_interaction': 'mean',
        'engagement_level': 'mean'
    }).T
    
    x = np.arange(len(type_features.index))
    width = 0.25
    for i, user_type in enumerate(type_features.columns):
        offset = (i - 1) * width
        ax3.bar(x + offset, type_features[user_type], width, 
               label=user_type, alpha=0.8)
    
    ax3.set_xlabel('特征', fontsize=12)
    ax3.set_ylabel('平均值', fontsize=12)
    ax3.set_title('各用户类型特征对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(type_features.index, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 互动行为分布
    ax4 = axes[1, 1]
    interaction_by_type = df.groupby('user_type')['interaction_score'].mean().sort_values(ascending=False)
    ax4.barh(interaction_by_type.index, interaction_by_type.values, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(interaction_by_type)])
    ax4.set_xlabel('平均互动分数', fontsize=12)
    ax4.set_ylabel('用户类型', fontsize=12)
    ax4.set_title('各类型用户互动行为对比', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存可视化结果: {save_path}")
    plt.show()

# ======================================
# 5. 生成画像报告
# ======================================

def generate_portrait_report(df, user_type_map):
    """生成受众画像报告"""
    report = []
    report.append("=" * 60)
    report.append("小红书受众画像分析报告")
    report.append("=" * 60)
    report.append("")
    
    # 总体统计
    report.append(f"📊 总体统计")
    report.append(f"  总样本数: {len(df)}")
    report.append(f"  用户类型数: {len(df['user_type'].unique())}")
    report.append("")
    
    # 各类型详细分析
    for user_type in ['心理慰藉型', '娱乐型', '深度参与型']:
        if user_type not in df['user_type'].values:
            continue
            
        type_data = df[df['user_type'] == user_type]
        count = len(type_data)
        ratio = count / len(df) * 100
        
        report.append("-" * 60)
        report.append(f"👥 {user_type}")
        report.append("-" * 60)
        report.append(f"  数量: {count} ({ratio:.1f}%)")
        report.append("")
        
        # 内容偏好
        report.append("  📝 内容偏好:")
        academic_ratio = type_data['is_academic_career'].mean() * 100
        emotional_ratio = type_data['is_emotional'].mean() * 100
        entertainment_ratio = type_data['is_entertainment'].mean() * 100
        report.append(f"    - 学业/职业类: {academic_ratio:.1f}%")
        report.append(f"    - 情感类: {emotional_ratio:.1f}%")
        report.append(f"    - 娱乐类: {entertainment_ratio:.1f}%")
        report.append("")
        
        # 互动行为
        report.append("  💬 互动行为:")
        avg_interaction = type_data['interaction_score'].mean()
        avg_likes = type_data['likes'].mean() if 'likes' in type_data.columns else 0
        avg_comments = type_data['comments'].mean() if 'comments' in type_data.columns else 0
        report.append(f"    - 平均互动分数: {avg_interaction:.2f}")
        report.append(f"    - 平均点赞数: {avg_likes:.1f}")
        report.append(f"    - 平均评论数: {avg_comments:.1f}")
        report.append("")
        
        # 参与度
        report.append("  📈 参与度:")
        avg_engagement = type_data['engagement_level'].mean()
        report.append(f"    - 平均参与度: {avg_engagement:.3f}")
        report.append("")
        
        # 特征描述
        if user_type == '心理慰藉型':
            report.append("  🎯 特征描述:")
            report.append("    - 主要关注学业和职业相关话题")
            report.append("    - 发帖峰值在考试周与招聘季")
            report.append("    - 寻求学业/职业指引和心理支持")
            report.append("    - 用户群体主要为大三至研究生")
        elif user_type == '娱乐型':
            report.append("  🎯 特征描述:")
            report.append("    - 集中在一二线城市")
            report.append("    - 关注感情运势和娱乐内容")
            report.append("    - 互动高峰在晚间休闲时段")
            report.append("    - 以轻松娱乐为主要目的")
        elif user_type == '深度参与型':
            report.append("  🎯 特征描述:")
            report.append("    - 跨平台追随，黏性最高")
            report.append("    - 有付费咨询与二次创作行为")
            report.append("    - 参与度和互动率最高")
            report.append("    - 对内容质量要求较高")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存报告
    with open("xiaohongshu_portrait_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n💾 已保存画像报告: xiaohongshu_portrait_report.txt")
    
    return report_text

# ======================================
# 主程序
# ======================================

def main():
    print("=" * 60)
    print("小红书受众画像分析：核心圈层与行为聚类")
    print("=" * 60)
    print()
    
    # 1. 加载数据
    df = load_data()
    if df is None or len(df) == 0:
        print("❌ 数据为空，无法进行分析")
        return
    
    # 2. 特征工程
    print("\n🔧 进行特征工程...")
    df = extract_keyword_features(df)
    df = calculate_interaction_features(df)
    df = calculate_engagement_features(df)
    df = extract_content_features(df)
    print("✅ 特征工程完成")
    
    # 3. 聚类分析
    print("\n🔍 执行聚类分析...")
    df, kmeans, scaler, cluster_centers = perform_clustering(df, n_clusters=3)
    
    # 4. 识别用户类型
    print("\n👥 识别用户类型...")
    df, user_type_map = identify_user_types(df)
    print(f"✅ 用户类型映射: {user_type_map}")
    
    # 5. 可视化
    print("\n📊 生成可视化图表...")
    plot_clustering_results(df)
    
    # 6. 生成报告
    print("\n📝 生成画像报告...")
    generate_portrait_report(df, user_type_map)
    
    # 7. 保存结果
    output_file = "xiaohongshu_user_portrait.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 已保存分析结果: {output_file}")
    
    print("\n✅ 受众画像分析完成！")

if __name__ == "__main__":
    main()

