# ======================================
# 聚类结果诊断分析（简化版，避免内存问题）
# ======================================

import sys
import os

# 分步导入，避免内存问题
try:
    import json
    print("✅ json 导入成功")
except Exception as e:
    print(f"❌ json 导入失败: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✅ pandas 导入成功")
except Exception as e:
    print(f"❌ pandas 导入失败: {e}")
    print("请尝试: pip install --upgrade pandas numpy")
    sys.exit(1)

try:
    import numpy as np
    print("✅ numpy 导入成功")
except Exception as e:
    print(f"❌ numpy 导入失败: {e}")
    print("请尝试: pip install --upgrade numpy")
    sys.exit(1)

# 延迟导入主模块
print("\n正在导入分析模块...")
try:
    from user_portrait_analysis import (
        load_data, standardize_columns, extract_time_features,
        extract_content_features, calculate_interaction_features,
        calculate_user_engagement_features, extract_sentiment_features,
        perform_clustering, identify_user_types
    )
    print("✅ 分析模块导入成功")
except Exception as e:
    print(f"❌ 分析模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 加载数据
print("\n" + "=" * 60)
print("聚类结果诊断分析")
print("=" * 60)

df = load_data()
if df is None or len(df) == 0:
    print("❌ 数据为空")
    sys.exit(1)

print(f"✅ 数据加载成功，共 {len(df)} 条记录")

# 分步处理，避免内存问题
print("\n🔧 步骤1: 数据标准化...")
df = standardize_columns(df)

print("🔧 步骤2: 提取时间特征...")
df = extract_time_features(df)

print("🔧 步骤3: 提取内容特征...")
df = extract_content_features(df)

print("🔧 步骤4: 计算互动特征...")
df = calculate_interaction_features(df)

print("🔧 步骤5: 计算用户参与度特征...")
df = calculate_user_engagement_features(df)

print("🔧 步骤6: 提取情感特征...")
df = extract_sentiment_features(df)

# 检查时间特征
print("\n📊 时间特征检查:")
try:
    time_success = df['created_datetime'].notna().sum()
    print(f"  时间解析成功数: {time_success} / {len(df)} ({time_success/len(df)*100:.1f}%)")
    print(f"  考试周比例: {df['is_exam_season'].mean():.3f}")
    print(f"  招聘季比例: {df['is_recruitment_season'].mean():.3f}")
    print(f"  休闲时段比例: {df['is_leisure_time'].mean():.3f}")
except Exception as e:
    print(f"  ⚠️ 时间特征检查失败: {e}")

# 检查内容特征
print("\n📝 内容特征检查:")
try:
    print(f"  学业/职业类比例: {df['is_academic_career'].mean():.3f}")
    print(f"  情感类比例: {df['is_emotional'].mean():.3f}")
    print(f"  娱乐类比例: {df['is_entertainment'].mean():.3f}")
    print(f"  平均学业得分: {df['academic_score'].mean():.2f}")
    print(f"  平均职业得分: {df['career_score'].mean():.2f}")
    print(f"  平均慰藉得分: {df['comfort_score'].mean():.2f}")
    print(f"  平均慰藉需求: {df['comfort_need'].mean():.3f}")
except Exception as e:
    print(f"  ⚠️ 内容特征检查失败: {e}")

# 执行聚类
print("\n🔍 执行聚类分析...")
try:
    df, kmeans, scaler, cluster_centers = perform_clustering(df, n_clusters=3)
except Exception as e:
    print(f"  ❌ 聚类失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 分析每个簇的特征
print("\n🔍 各簇特征分析:")
for cluster_id in sorted(df['cluster'].unique()):
    try:
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"\n簇 {cluster_id} (样本数: {len(cluster_data)}):")
        print(f"  学业/职业类比例: {cluster_data['is_academic_career'].mean():.3f}")
        print(f"  情感类比例: {cluster_data['is_emotional'].mean():.3f}")
        print(f"  娱乐类比例: {cluster_data['is_entertainment'].mean():.3f}")
        print(f"  平均互动分数: {cluster_data['interaction_score'].mean():.2f}")
        print(f"  平均参与度: {cluster_data['engagement_level'].mean():.3f}")
        print(f"  平均活跃度: {cluster_data['activity_level'].mean():.3f}")
        print(f"  平均慰藉需求: {cluster_data['comfort_need'].mean():.3f}")
        print(f"  平均深度得分: {cluster_data['deep_score'].mean():.2f}")
        print(f"  考试周比例: {cluster_data['is_exam_season'].mean():.3f}")
        print(f"  招聘季比例: {cluster_data['is_recruitment_season'].mean():.3f}")
        print(f"  休闲时段比例: {cluster_data['is_leisure_time'].mean():.3f}")
    except Exception as e:
        print(f"  ⚠️ 簇 {cluster_id} 分析失败: {e}")

# 识别用户类型
print("\n👥 识别用户类型...")
try:
    df, user_type_map = identify_user_types(df)
    
    print("\n👥 用户类型识别结果:")
    print(f"  类型映射: {user_type_map}")
    print(f"\n  各类型数量:")
    type_counts = df['user_type'].value_counts()
    for user_type, count in type_counts.items():
        ratio = count / len(df) * 100
        print(f"    {user_type}: {count} ({ratio:.1f}%)")
except Exception as e:
    print(f"  ❌ 用户类型识别失败: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ 诊断分析完成！")

# 添加可视化
print("\n📊 生成可视化图表...")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 聚类散点图（PCA降维）
    ax1 = axes[0, 0]
    feature_cols = ['is_academic_career', 'is_emotional', 'is_entertainment', 
                   'log_interaction', 'interaction_diversity', 'engagement_level',
                   'comfort_score', 'deep_score']
    X = df[feature_cols].fillna(0)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    # 按用户类型着色
    type_colors = {'心理慰藉型': '#FF6B6B', '娱乐型': '#4ECDC4', '深度参与型': '#45B7D1'}
    colors = df['user_type'].map(type_colors).fillna('#999999')
    
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=50)
    ax1.set_xlabel('主成分1', fontsize=12)
    ax1.set_ylabel('主成分2', fontsize=12)
    ax1.set_title('用户聚类结果（PCA降维）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) 
                      for label, color in type_colors.items() 
                      if label in df['user_type'].values]
    if legend_elements:
        ax1.legend(handles=legend_elements, loc='best')
    
    # 2. 用户类型分布
    ax2 = axes[0, 1]
    user_type_counts = df['user_type'].value_counts()
    colors_list = [type_colors.get(ut, '#999999') for ut in user_type_counts.index]
    bars = ax2.bar(user_type_counts.index, user_type_counts.values, color=colors_list)
    ax2.set_xlabel('用户类型', fontsize=12)
    ax2.set_ylabel('数量', fontsize=12)
    ax2.set_title('三类用户群体分布', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # 3. 各类型特征对比
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
        offset = (i - len(type_features.columns)/2 + 0.5) * width
        ax3.bar(x + offset, type_features[user_type], width, 
               label=user_type, alpha=0.8, color=type_colors.get(user_type, '#999999'))
    
    ax3.set_xlabel('特征', fontsize=12)
    ax3.set_ylabel('平均值', fontsize=12)
    ax3.set_title('各用户类型特征对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(type_features.index, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 互动行为对比
    ax4 = axes[1, 1]
    interaction_by_type = df.groupby('user_type').agg({
        'interaction_score': 'mean',
        'reposts_count': 'mean',
        'comments_count': 'mean',
        'attitudes_count': 'mean'
    })
    
    x = np.arange(len(interaction_by_type.index))
    width = 0.2
    metrics = ['interaction_score', 'reposts_count', 'comments_count', 'attitudes_count']
    metric_labels = ['互动总分', '转发', '评论', '点赞']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - len(metrics)/2 + 0.5) * width
        values = interaction_by_type[metric].values
        # 归一化显示
        if metric == 'interaction_score':
            values_norm = values / (values.max() + 1) * 100
        else:
            values_norm = values / (values.max() + 1) * 100
        ax4.bar(x + offset, values_norm, width, label=label, alpha=0.8)
    
    ax4.set_xlabel('用户类型', fontsize=12)
    ax4.set_ylabel('归一化值', fontsize=12)
    ax4.set_title('各类型用户互动行为对比', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(interaction_by_type.index, rotation=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = "weibo_clustering_diagnosis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存可视化结果: {save_path}")
    plt.show()
    
except Exception as e:
    print(f"⚠️ 可视化生成失败: {e}")
    import traceback
    traceback.print_exc()



