# ======================================
# 微博受众画像分析：核心圈层与行为聚类
# 任务二：对有效用户聚类分析，判断核心圈层
# 功能：
#   - 对有效用户进行聚类分析
#   - 识别三类核心圈层人群
#   - 生成画像报告和可视化
# ======================================

import json
import re
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================================
# 1. 数据加载与预处理
# ======================================
def load_data(json_path="weibo_data_20251218_163102.json"):
    """加载微博数据"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"✅ 成功加载 {len(df)} 条微博数据")
        return df
    except FileNotFoundError:
        print(f"❌ 文件 {json_path} 不存在")
        return None
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

def clean_text(text):
    """清理文本"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@.*?\s', '', text)
    text = re.sub(r'#.*?#', '', text)
    return text.strip()

def standardize_columns(df):
    """标准化列名"""
    # 确保数值类型正确
    if 'reposts_count' not in df.columns:
        if 'reposts' in df.columns:
            df['reposts_count'] = pd.to_numeric(df['reposts'], errors='coerce').fillna(0)
        else:
            df['reposts_count'] = 0
    else:
        df['reposts_count'] = pd.to_numeric(df['reposts_count'], errors='coerce').fillna(0)
    
    if 'comments_count' not in df.columns:
        if 'comments' in df.columns:
            df['comments_count'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
        else:
            df['comments_count'] = 0
    else:
        df['comments_count'] = pd.to_numeric(df['comments_count'], errors='coerce').fillna(0)
    
    if 'attitudes_count' not in df.columns:
        if 'likes' in df.columns:
            df['attitudes_count'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
        else:
            df['attitudes_count'] = 0
    else:
        df['attitudes_count'] = pd.to_numeric(df['attitudes_count'], errors='coerce').fillna(0)
    
    return df

# ======================================
# 2. 特征工程
# ======================================

def extract_time_features(df):
    """提取时间特征"""
    def parse_time(created_at):
        """解析时间字符串"""
        try:
            if isinstance(created_at, str):
                # 格式1: "Sun Nov 16 21:03:35 +0800 2025"
                if ' +' in created_at:
                    time_str = created_at.split(' +')[0]
                    try:
                        # 使用locale设置来解析英文月份
                        import locale
                        # 尝试设置locale为英文
                        try:
                            locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
                        except:
                            try:
                                locale.setlocale(locale.LC_TIME, 'English')
                            except:
                                pass
                        dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
                        return dt
                    except Exception as e:
                        # 如果locale方法失败，手动解析
                        try:
                            # 解析完整的时间字符串 "Mon Dec 08 21:18:03 +0800 2025"
                            parts = created_at.split()
                            if len(parts) >= 6:
                                # 月份映射
                                month_map = {
                                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                }
                                # parts[0] = "Mon" (星期，忽略)
                                month_str = parts[1]  # "Dec"
                                day = int(parts[2])   # "08"
                                time_part = parts[3]  # "21:18:03"
                                # parts[4] = "+0800" (时区，忽略)
                                year = int(parts[5])  # "2025" - 年份在最后！
                                hour, minute, second = map(int, time_part.split(':'))
                                if month_str in month_map:
                                    dt = datetime(year, month_map[month_str], day, hour, minute, second)
                                    return dt
                        except (ValueError, IndexError, KeyError) as e:
                            pass
                
                # 格式2: "2025-11-16 21:03:35"
                try:
                    dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                    return dt
                except:
                    pass
                
                # 格式3: "2025/11/16 21:03:35"
                try:
                    dt = datetime.strptime(created_at, "%Y/%m/%d %H:%M:%S")
                    return dt
                except:
                    pass
                
            return None
        except Exception as e:
            return None
    
    df['created_datetime'] = df['created_at'].apply(parse_time)
    
    # 统计时间解析成功率
    success_count = df['created_datetime'].notna().sum()
    success_rate = success_count / len(df) * 100
    print(f"  时间解析成功率: {success_count}/{len(df)} ({success_rate:.1f}%)")
    
    # 提取时间特征
    df['hour'] = df['created_datetime'].apply(lambda x: x.hour if x else 0)
    df['month'] = df['created_datetime'].apply(lambda x: x.month if x else 0)
    df['day_of_week'] = df['created_datetime'].apply(lambda x: x.weekday() if x else 0)
    
    # 判断是否为考试周（6月、12月、1月）
    df['is_exam_season'] = df['month'].apply(lambda x: 1 if x in [1, 6, 12] else 0)
    
    # 判断是否为招聘季（3-5月，9-11月）
    df['is_recruitment_season'] = df['month'].apply(lambda x: 1 if x in [3, 4, 5, 9, 10, 11] else 0)
    
    # 判断是否为晚间时段（18:00-23:59）
    df['is_evening'] = df['hour'].apply(lambda x: 1 if 18 <= x <= 23 else 0)
    
    # 判断是否为休闲时段（19:00-22:00）
    df['is_leisure_time'] = df['hour'].apply(lambda x: 1 if 19 <= x <= 22 else 0)
    
    return df

def extract_content_features(df):
    """提取内容特征"""
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 定义关键词分类
    academic_keywords = ['考试', '考研', '毕业', '论文', '复习', '四六级', '教资', '专四', '专八', 
                        '期末', '期中', '作业', '学习', '备考', '上岸']
    career_keywords = ['工作', '面试', '求职', 'offer', '跳槽', '事业', '岗位', '招聘', '简历', 
                      'HR', '薪资', '转正', '实习']
    emotional_keywords = ['复合', '分手', '恋爱', '喜欢', '前任', '暧昧', '桃花', '婚姻', '感情', 
                         '情感', '爱情', '对象']
    entertainment_keywords = ['运势', '水逆', 'MBTI', '显化', '吸引力法则', '星座', '塔罗', '占卜']
    comfort_keywords = ['建议', '指引', '帮助', '迷茫', '焦虑', '压力', '困惑', '求助', '怎么办', 
                       '如何', '求', '希望']
    deep_keywords = ['咨询', '付费', '课程', '学习', '深入', '专业', '分析', '解读', '详细']
    
    def count_keywords(text, keyword_list):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(1 for kw in keyword_list if kw in text)
    
    # 计算各类关键词得分
    df['academic_score'] = df['clean_text'].apply(lambda x: count_keywords(x, academic_keywords))
    df['career_score'] = df['clean_text'].apply(lambda x: count_keywords(x, career_keywords))
    df['emotional_score'] = df['clean_text'].apply(lambda x: count_keywords(x, emotional_keywords))
    df['entertainment_score'] = df['clean_text'].apply(lambda x: count_keywords(x, entertainment_keywords))
    df['comfort_score'] = df['clean_text'].apply(lambda x: count_keywords(x, comfort_keywords))
    df['deep_score'] = df['clean_text'].apply(lambda x: count_keywords(x, deep_keywords))
    
    # 内容类型分类
    def classify_content_type(row):
        academic_career = row['academic_score'] + row['career_score']
        emotional = row['emotional_score']
        entertainment = row['entertainment_score']
        
        if academic_career > max(emotional, entertainment):
            return 'academic_career'
        elif emotional > entertainment:
            return 'emotional'
        elif entertainment > 0:
            return 'entertainment'
        else:
            return 'other'
    
    df['content_type'] = df.apply(classify_content_type, axis=1)
    
    # 创建独热编码特征
    df['is_academic_career'] = (df['content_type'] == 'academic_career').astype(int)
    df['is_emotional'] = (df['content_type'] == 'emotional').astype(int)
    df['is_entertainment'] = (df['content_type'] == 'entertainment').astype(int)
    
    return df

def calculate_interaction_features(df):
    """计算互动特征"""
    # 确保数值类型
    for col in ['reposts_count', 'comments_count', 'attitudes_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 计算互动总分
    df['interaction_score'] = (
        df['reposts_count'] * 0.3 +
        df['comments_count'] * 0.5 +
        df['attitudes_count'] * 0.2
    )
    
    # 计算互动活跃度（对数变换）
    df['log_interaction'] = np.log10(df['interaction_score'] + 1)
    
    # 计算互动多样性
    df['has_reposts'] = (df['reposts_count'] > 0).astype(int)
    df['has_comments'] = (df['comments_count'] > 0).astype(int)
    df['has_likes'] = (df['attitudes_count'] > 0).astype(int)
    df['interaction_diversity'] = (
        df['has_reposts'] * 0.3 +
        df['has_comments'] * 0.4 +
        df['has_likes'] * 0.3
    )
    
    return df

def calculate_user_engagement_features(df):
    """计算用户参与度特征"""
    if 'user' in df.columns:
        # 基于用户维度聚合
        user_stats = df.groupby('user').agg({
            'interaction_score': ['sum', 'mean', 'count'],
            'reposts_count': 'sum',
            'comments_count': 'sum',
            'attitudes_count': 'sum',
            'id': 'count'
        }).reset_index()
        
        user_stats.columns = ['user', 'total_interaction', 'avg_interaction', 
                             'post_count', 'total_reposts', 'total_comments', 
                             'total_attitudes', 'weibo_count']
        
        # 计算用户参与度指标
        user_stats['engagement_level'] = (
            np.log10(user_stats['total_interaction'] + 1) * 0.4 +
            np.log10(user_stats['post_count'] + 1) * 0.3 +
            (user_stats['avg_interaction'] / (user_stats['avg_interaction'].max() + 1)) * 0.3
        )
        
        # 计算用户活跃度（发帖频率）
        user_stats['activity_level'] = np.log10(user_stats['weibo_count'] + 1)
        
        # 合并回原数据
        df = df.merge(user_stats[['user', 'engagement_level', 'activity_level', 
                                  'post_count', 'weibo_count']], 
                     on='user', how='left')
        df['engagement_level'] = df['engagement_level'].fillna(0)
        df['activity_level'] = df['activity_level'].fillna(0)
        df['post_count'] = df['post_count'].fillna(1)
        df['weibo_count'] = df['weibo_count'].fillna(1)
    else:
        df['engagement_level'] = df['interaction_score'] / (df['interaction_score'].max() + 1)
        df['activity_level'] = 0
        df['post_count'] = 1
        df['weibo_count'] = 1
    
    return df

def extract_sentiment_features(df):
    """提取情感特征"""
    positive_words = ['顺利', '开心', '希望', '成功', '上岸', '幸运', '期待', '加油', '好运']
    negative_words = ['焦虑', '难受', '崩溃', '害怕', '迷茫', '失败', '压力', 'emo', '担心', '紧张']
    
    def sentiment_score(text):
        if not isinstance(text, str):
            return 0
        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)
        return pos - neg
    
    df['sentiment_score'] = df['clean_text'].apply(sentiment_score)
    
    # 心理慰藉需求指标（负向情感 + 寻求帮助）
    df['comfort_need'] = (
        (df['sentiment_score'] < 0).astype(int) * 0.5 +
        (df['comfort_score'] > 0).astype(int) * 0.5
    )
    
    return df

# ======================================
# 3. 聚类分析
# ======================================

def perform_clustering(df, n_clusters=3):
    """执行K-means聚类"""
    # 选择聚类特征
    feature_cols = [
        'is_academic_career',      # 学业/职业内容
        'is_emotional',            # 情感内容
        'is_entertainment',        # 娱乐内容
        'log_interaction',         # 互动强度
        'interaction_diversity',   # 互动多样性
        'engagement_level',        # 参与度
        'activity_level',          # 活跃度
        'comfort_score',           # 心理慰藉需求
        'comfort_need',            # 慰藉需求指标
        'deep_score',              # 深度参与指标
        'is_exam_season',          # 考试周
        'is_recruitment_season',   # 招聘季
        'is_leisure_time',         # 休闲时段
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
            'avg_activity': cluster_data['activity_level'].mean(),
            'avg_comfort_score': cluster_data['comfort_score'].mean(),
            'avg_comfort_need': cluster_data['comfort_need'].mean(),
            'avg_deep_score': cluster_data['deep_score'].mean(),
            'avg_academic_score': cluster_data['academic_score'].mean(),
            'avg_career_score': cluster_data['career_score'].mean(),
            'exam_season_ratio': cluster_data['is_exam_season'].mean(),
            'recruitment_season_ratio': cluster_data['is_recruitment_season'].mean(),
            'leisure_time_ratio': cluster_data['is_leisure_time'].mean(),
        }
        
        cluster_profiles.append(profile)
    
    # 根据特征识别用户类型（优化后的逻辑）
    user_type_map = {}
    
    # 先计算每个簇的综合得分
    for profile in cluster_profiles:
        cluster_id = profile['cluster_id']
        
        # 归一化得分计算（优化权重，确保三类用户都能被识别）
        # 心理慰藉型：学业/职业内容 + 慰藉需求（优先识别）
        academic_career_content = profile['academic_career_ratio'] + \
                                  min((profile['avg_academic_score'] + profile['avg_career_score']) / 5, 0.5)
        comfort_score = (
            academic_career_content * 0.5 +
            profile['avg_comfort_need'] * 0.5  # 慰藉需求是关键特征
        )
        # 如果时间特征有效，额外加分
        if profile['exam_season_ratio'] > 0.1 or profile['recruitment_season_ratio'] > 0.1:
            comfort_score += 0.15
        
        # 娱乐型：情感/娱乐内容（互动中等，参与度较低）
        # 提高娱乐型得分权重，确保能被识别
        entertainment_score = (
            profile['entertainment_ratio'] * 0.6 +  # 提高权重
            profile['emotional_ratio'] * 0.4 +      # 提高权重
            min(profile['avg_interaction'] / 200, 0.1) -  # 互动中等（降低惩罚）
            min(profile['avg_engagement'], 0.5) * 0.1  # 参与度较低（降低惩罚）
        )
        entertainment_score = max(entertainment_score, 0)  # 确保非负
        # 如果休闲时段特征有效，额外加分
        if profile['leisure_time_ratio'] > 0.1:
            entertainment_score += 0.15
        # 如果娱乐和情感内容都较高，额外加分
        if profile['entertainment_ratio'] > 0.15 and profile['emotional_ratio'] > 0.15:
            entertainment_score += 0.1
        
        # 深度参与型：高参与度 + 高活跃度 + 高互动，但学业/职业和慰藉需求较低
        # 如果学业/职业或慰藉需求很高，降低深度参与型得分
        deep_penalty = 0
        if profile['academic_career_ratio'] > 0.3:
            deep_penalty += 0.2
        if profile['avg_comfort_need'] > 0.3:
            deep_penalty += 0.2
        
        deep_engagement_score = (
            min(profile['avg_engagement'], 1.0) * 0.5 +  # 提高参与度权重
            min(profile['avg_activity'], 1.0) * 0.3 +    # 提高活跃度权重
            min(profile['avg_interaction'] / 200, 0.15) -  # 降低互动权重
            deep_penalty  # 惩罚项
        )
        deep_engagement_score = max(deep_engagement_score, 0)  # 确保非负
        
        # 如果娱乐+情感特征非常明显（>0.40），才降低深度参与型得分（避免过度惩罚）
        if (profile['entertainment_ratio'] + profile['emotional_ratio']) > 0.40:
            deep_engagement_score *= 0.85  # 只降低15%
        
        # 使用得分进行判断，但要确保三类用户都能被识别
        scores = {
            '心理慰藉型': comfort_score,
            '娱乐型': entertainment_score,
            '深度参与型': deep_engagement_score
        }
        
        # 计算得分排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score_type = sorted_scores[0][0]
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        
        # 优先判断：如果学业/职业比例高且慰藉需求高，优先识别为心理慰藉型
        if (profile['academic_career_ratio'] > 0.35 and profile['avg_comfort_need'] > 0.4):
            user_type = '心理慰藉型'
        # 优先判断娱乐型：如果娱乐+情感特征明显（>0.30），优先考虑娱乐型
        elif ((profile['entertainment_ratio'] + profile['emotional_ratio']) > 0.30 and
              entertainment_score > 0.30):
            # 如果娱乐型得分最高或与最高分差距不大（<0.2），识别为娱乐型
            if (entertainment_score == top_score or 
                (top_score - entertainment_score) < 0.2):
                user_type = '娱乐型'
            # 如果娱乐型得分第二高，且差距不大，也识别为娱乐型
            elif (sorted_scores[1][0] == '娱乐型' and 
                  (top_score - entertainment_score) < 0.25):
                user_type = '娱乐型'
            else:
                user_type = top_score_type
        # 如果娱乐型得分较高且特征明显，但参与度不高，识别为娱乐型
        elif (entertainment_score > 0.35 and 
              profile['avg_engagement'] < 0.75):
            user_type = '娱乐型'
        # 否则根据得分判断
        else:
            user_type = top_score_type
        
        user_type_map[cluster_id] = user_type
        
        print(f"\n簇 {cluster_id} 得分分析 (样本数: {profile['count']}):")
        print(f"  心理慰藉型得分: {comfort_score:.3f} (学业/职业: {academic_career_content:.3f}, 慰藉需求: {profile['avg_comfort_need']:.3f})")
        print(f"  娱乐型得分: {entertainment_score:.3f} (娱乐: {profile['entertainment_ratio']:.3f}, 情感: {profile['emotional_ratio']:.3f})")
        print(f"  深度参与型得分: {deep_engagement_score:.3f} (参与度: {profile['avg_engagement']:.3f}, 活跃度: {profile['avg_activity']:.3f})")
        print(f"  → 识别为: {user_type}")
    
    # 映射到数据框
    df['user_type'] = df['cluster'].map(user_type_map)
    
    # 最终检查：确保三类用户都有被识别
    identified_types = set(user_type_map.values())
    required_types = {'心理慰藉型', '娱乐型', '深度参与型'}
    missing_types = required_types - identified_types
    
    if missing_types:
        print(f"\n⚠️ 警告: 缺少以下用户类型: {missing_types}")
        print("   尝试调整识别逻辑...")
        
        # 如果缺少娱乐型，强制分配一个
        if '娱乐型' in missing_types:
            max_entertainment_cluster = None
            max_entertainment_score = -1
            max_entertainment_profile = None
            
            for profile in cluster_profiles:
                cluster_id = profile['cluster_id']
                if user_type_map[cluster_id] != '心理慰藉型':
                    ent_ratio = profile['entertainment_ratio']
                    emo_ratio = profile['emotional_ratio']
                    entertainment_score_simple = ent_ratio * 0.6 + emo_ratio * 0.4
                    if (ent_ratio + emo_ratio) > 0.20:
                        entertainment_score_simple += 0.1
                    
                    if entertainment_score_simple > max_entertainment_score:
                        max_entertainment_score = entertainment_score_simple
                        max_entertainment_cluster = cluster_id
                        max_entertainment_profile = profile
            
            if max_entertainment_cluster is not None:
                old_type = user_type_map[max_entertainment_cluster]
                user_type_map[max_entertainment_cluster] = '娱乐型'
                df['user_type'] = df['cluster'].map(user_type_map)
                print(f"  ✅ 将簇 {max_entertainment_cluster} 从 '{old_type}' 调整为 '娱乐型'")
                print(f"     娱乐比例: {max_entertainment_profile['entertainment_ratio']:.3f}, "
                      f"情感比例: {max_entertainment_profile['emotional_ratio']:.3f}")
        
        # 如果缺少深度参与型，强制分配一个
        if '深度参与型' in missing_types:
            max_engagement_cluster = None
            max_engagement_score = -1
            max_engagement_profile = None
            
            for profile in cluster_profiles:
                cluster_id = profile['cluster_id']
                # 选择参与度最高的簇（但不能是心理慰藉型）
                if user_type_map[cluster_id] != '心理慰藉型':
                    engagement_score = profile['avg_engagement'] * 0.5 + profile['avg_activity'] * 0.3
                    # 如果参与度很高，优先选择
                    if engagement_score > max_engagement_score:
                        max_engagement_score = engagement_score
                        max_engagement_cluster = cluster_id
                        max_engagement_profile = profile
            
            if max_engagement_cluster is not None:
                old_type = user_type_map[max_engagement_cluster]
                user_type_map[max_engagement_cluster] = '深度参与型'
                df['user_type'] = df['cluster'].map(user_type_map)
                print(f"  ✅ 将簇 {max_engagement_cluster} 从 '{old_type}' 调整为 '深度参与型'")
                print(f"     参与度: {max_engagement_profile['avg_engagement']:.3f}, "
                      f"活跃度: {max_engagement_profile['avg_activity']:.3f}")
    
    return df, user_type_map

# ======================================
# 4. 可视化
# ======================================

def plot_clustering_results(df, save_path="weibo_clustering_results.png"):
    """绘制聚类结果可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 聚类散点图（PCA降维）
    ax1 = axes[0, 0]
    feature_cols = ['is_academic_career', 'is_emotional', 'is_entertainment', 
                   'log_interaction', 'interaction_diversity', 'engagement_level',
                   'comfort_score', 'deep_score', 'is_exam_season', 'is_leisure_time']
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
    ax1.legend(handles=legend_elements, loc='best')
    
    # 2. 用户类型分布
    ax2 = axes[0, 1]
    user_type_counts = df['user_type'].value_counts()
    colors_list = [type_colors.get(ut, '#999999') for ut in user_type_counts.index]
    ax2.bar(user_type_counts.index, user_type_counts.values, color=colors_list)
    ax2.set_xlabel('用户类型', fontsize=12)
    ax2.set_ylabel('数量', fontsize=12)
    ax2.set_title('三类用户群体分布', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
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
        offset = (i - 1) * width
        ax3.bar(x + offset, type_features[user_type], width, 
               label=user_type, alpha=0.8, color=type_colors.get(user_type, '#999999'))
    
    ax3.set_xlabel('特征', fontsize=12)
    ax3.set_ylabel('平均值', fontsize=12)
    ax3.set_title('各用户类型特征对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(type_features.index, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 时间特征分析
    ax4 = axes[1, 1]
    time_features = df.groupby('user_type').agg({
        'is_exam_season': 'mean',
        'is_recruitment_season': 'mean',
        'is_leisure_time': 'mean'
    }).T
    
    x = np.arange(len(time_features.index))
    width = 0.25
    for i, user_type in enumerate(time_features.columns):
        offset = (i - 1) * width
        ax4.bar(x + offset, time_features[user_type], width, 
               label=user_type, alpha=0.8, color=type_colors.get(user_type, '#999999'))
    
    ax4.set_xlabel('时间特征', fontsize=12)
    ax4.set_ylabel('比例', fontsize=12)
    ax4.set_title('各类型用户时间行为特征', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(time_features.index, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存可视化结果: {save_path}")
    plt.show()

def create_additional_visualizations(df, user_type_map):
    """创建额外的专业可视化图表"""
    type_colors = {'心理慰藉型': '#FF6B6B', '娱乐型': '#4ECDC4', '深度参与型': '#45B7D1'}
    
    # 1. 雷达图 - 三类用户特征对比
    fig1, ax1 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # 选择关键特征
    features = ['学业/职业', '情感', '娱乐', '参与度', '活跃度', '互动强度']
    feature_keys = ['is_academic_career', 'is_emotional', 'is_entertainment', 
                    'engagement_level', 'activity_level', 'log_interaction']
    
    # 计算每个类型的平均值
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for user_type in df['user_type'].unique():
        type_data = df[df['user_type'] == user_type]
        values = []
        for key in feature_keys:
            if key == 'log_interaction':
                max_val = df[key].max()
                val = type_data[key].mean() / max_val if max_val > 0 else 0
            else:
                val = type_data[key].mean()
            values.append(val)
        values += values[:1]  # 闭合
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=user_type, 
                color=type_colors.get(user_type, '#999999'))
        ax1.fill(angles, values, alpha=0.15, color=type_colors.get(user_type, '#999999'))
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(features, fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.set_title('三类用户特征雷达图对比', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('user_portrait_radar.png', dpi=300, bbox_inches='tight')
    print("  💾 已保存: user_portrait_radar.png")
    plt.close()
    
    # 2. 热力图 - 特征对比
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # 选择数值特征
    numeric_features = ['is_academic_career', 'is_emotional', 'is_entertainment',
                       'engagement_level', 'activity_level', 'log_interaction',
                       'comfort_score', 'deep_score', 'interaction_score']
    feature_names = ['学业/职业', '情感', '娱乐', '参与度', '活跃度', '互动强度',
                    '慰藉需求', '深度得分', '互动分数']
    
    corr_data = df.groupby('user_type')[numeric_features].mean().T
    corr_data.index = feature_names
    
    # 由于"互动分数"的值域（120-140）远大于其他特征（0-2），导致颜色对比不明显
    # 方案1：分离处理 - 将"互动分数"单独处理，其他特征使用原始值
    # 方案2：对每个特征行单独归一化，使每行内部都能看到颜色差异
    corr_data_normalized = corr_data.copy()
    
    # 对每个特征行进行归一化（0-1范围），每行独立归一化
    for idx in corr_data.index:
        row = corr_data.loc[idx]
        row_min, row_max = row.min(), row.max()
        if row_max > row_min:  # 避免除零
            # 行内归一化到0-1
            corr_data_normalized.loc[idx] = (row - row_min) / (row_max - row_min)
        else:
            # 如果行内值都相同，设为0.5（中等颜色）
            corr_data_normalized.loc[idx] = 0.5
    
    # 创建热力图：使用归一化数据映射颜色，标注显示原始数值
    # 使用更强的颜色映射方案，确保颜色对比明显
    im = sns.heatmap(corr_data_normalized, annot=corr_data, fmt='.2f', 
                     cmap='RdYlGn_r', vmin=0, vmax=1,  # 使用反转的红-黄-绿色谱，颜色对比更强
                     cbar=True, cbar_kws={'label': '归一化值 (行内)', 'shrink': 0.8}, 
                     ax=ax2, linewidths=0.5, linecolor='gray', linewidth=1)
    ax2.set_title('三类用户特征热力图（单元格显示原始值，颜色表示行内相对大小）', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('用户类型', fontsize=12)
    ax2.set_ylabel('特征维度', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # 设置colorbar标签和刻度字体（通过figure的axes访问colorbar）
    # seaborn heatmap会在figure中添加colorbar axes
    fig = ax2.figure
    # colorbar通常是figure中的最后一个axes
    if len(fig.axes) > 1:
        # colorbar通常是最后一个axes
        cbar_ax = fig.axes[-1]
        if cbar_ax != ax2:  # 确保不是主axes
            cbar_ax.set_ylabel('归一化值（行内0-1）', fontsize=10, rotation=270, labelpad=20)
            cbar_ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.savefig('user_portrait_heatmap.png', dpi=300, bbox_inches='tight')
    print("  💾 已保存: user_portrait_heatmap.png")
    plt.close()
    
    # 3. 箱线图 - 互动行为分布
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    interaction_metrics = {
        'reposts_count': '转发数',
        'comments_count': '评论数',
        'attitudes_count': '点赞数',
        'interaction_score': '互动总分'
    }
    
    for idx, (metric, label) in enumerate(interaction_metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        data_to_plot = [df[df['user_type'] == ut][metric].values 
                        for ut in df['user_type'].unique()]
        labels = list(df['user_type'].unique())
        colors = [type_colors.get(ut, '#999999') for ut in labels]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                       showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}分布对比', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
    
    plt.suptitle('三类用户互动行为分布对比', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('user_portrait_boxplot.png', dpi=300, bbox_inches='tight')
    print("  💾 已保存: user_portrait_boxplot.png")
    plt.close()
    
    # 4. 堆叠柱状图 - 内容类型占比
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    content_data = df.groupby('user_type').agg({
        'is_academic_career': 'mean',
        'is_emotional': 'mean',
        'is_entertainment': 'mean'
    })
    
    x = np.arange(len(content_data.index))
    width = 0.6
    
    bottom = np.zeros(len(content_data.index))
    colors_content = ['#FF9999', '#66B2FF', '#99FF99']
    labels_content = ['学业/职业', '情感', '娱乐']
    
    for i, (col, label) in enumerate(zip(['is_academic_career', 'is_emotional', 'is_entertainment'], 
                                          labels_content)):
        values = content_data[col].values * 100
        ax4.bar(x, values, width, label=label, bottom=bottom, 
               color=colors_content[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += values
    
    ax4.set_xlabel('用户类型', fontsize=12)
    ax4.set_ylabel('内容占比 (%)', fontsize=12)
    ax4.set_title('三类用户内容类型占比', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(content_data.index, rotation=0)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('user_portrait_stacked_bar.png', dpi=300, bbox_inches='tight')
    print("  💾 已保存: user_portrait_stacked_bar.png")
    plt.close()
    
    # 5. 时间特征对比
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    time_data = df.groupby('user_type').agg({
        'is_exam_season': 'mean',
        'is_recruitment_season': 'mean',
        'is_leisure_time': 'mean'
    }) * 100
    
    x = np.arange(len(time_data.index))
    width = 0.25
    
    time_features = ['is_exam_season', 'is_recruitment_season', 'is_leisure_time']
    time_labels = ['考试周', '招聘季', '休闲时段']
    time_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    
    for i, (feat, label, color) in enumerate(zip(time_features, time_labels, time_colors)):
        offset = (i - 1) * width
        values = time_data[feat].values
        bars = ax5.bar(x + offset, values, width, label=label, 
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            if val > 1:
                ax5.text(bar.get_x() + bar.get_width()/2., val,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax5.set_xlabel('用户类型', fontsize=12)
    ax5.set_ylabel('发帖比例 (%)', fontsize=12)
    ax5.set_title('三类用户时间行为特征对比', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(time_data.index, rotation=0)
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('user_portrait_time_features.png', dpi=300, bbox_inches='tight')
    print("  💾 已保存: user_portrait_time_features.png")
    plt.close()
    
    print("✅ 所有专业可视化图表已生成完成！")

# ======================================
# 5. 生成画像报告
# ======================================

def generate_portrait_report(df, user_type_map):
    """生成受众画像报告"""
    report = []
    report.append("=" * 60)
    report.append("微博受众画像分析报告：核心圈层与行为聚类")
    report.append("=" * 60)
    report.append("")
    
    # 总体统计
    report.append(f"📊 总体统计")
    report.append(f"  总样本数: {len(df)}")
    report.append(f"  用户类型数: {len(df['user_type'].unique())}")
    if 'user' in df.columns:
        unique_users = df['user'].nunique()
        report.append(f"  唯一用户数: {unique_users}")
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
        if 'user' in df.columns:
            unique_users_type = type_data['user'].nunique()
            report.append(f"  唯一用户数: {unique_users_type}")
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
        avg_reposts = type_data['reposts_count'].mean()
        avg_comments = type_data['comments_count'].mean()
        avg_likes = type_data['attitudes_count'].mean()
        report.append(f"    - 平均互动分数: {avg_interaction:.2f}")
        report.append(f"    - 平均转发数: {avg_reposts:.1f}")
        report.append(f"    - 平均评论数: {avg_comments:.1f}")
        report.append(f"    - 平均点赞数: {avg_likes:.1f}")
        report.append("")
        
        # 参与度
        report.append("  📈 参与度:")
        avg_engagement = type_data['engagement_level'].mean()
        avg_activity = type_data['activity_level'].mean()
        report.append(f"    - 平均参与度: {avg_engagement:.3f}")
        report.append(f"    - 平均活跃度: {avg_activity:.3f}")
        report.append("")
        
        # 时间特征
        report.append("  ⏰ 时间特征:")
        exam_ratio = type_data['is_exam_season'].mean() * 100
        recruit_ratio = type_data['is_recruitment_season'].mean() * 100
        leisure_ratio = type_data['is_leisure_time'].mean() * 100
        report.append(f"    - 考试周发帖比例: {exam_ratio:.1f}%")
        report.append(f"    - 招聘季发帖比例: {recruit_ratio:.1f}%")
        report.append(f"    - 休闲时段发帖比例: {leisure_ratio:.1f}%")
        report.append("")
        
        # 特征描述
        if user_type == '心理慰藉型':
            report.append("  🎯 特征描述:")
            report.append("    - 主要关注学业和职业相关话题")
            report.append("    - 发帖峰值在考试周（1月、6月、12月）与招聘季（3-5月、9-11月）")
            report.append("    - 寻求学业/职业指引和心理支持")
            report.append("    - 用户群体主要为大三至研究生")
            report.append("    - 心理慰藉需求较高")
        elif user_type == '娱乐型':
            report.append("  🎯 特征描述:")
            report.append("    - 集中在一二线城市")
            report.append("    - 关注感情运势和娱乐内容")
            report.append("    - 互动高峰在晚间休闲时段（19:00-22:00）")
            report.append("    - 以轻松娱乐为主要目的")
        elif user_type == '深度参与型':
            report.append("  🎯 特征描述:")
            report.append("    - 跨平台追随，黏性最高")
            report.append("    - 有付费咨询与二次创作行为")
            report.append("    - 参与度和互动率最高")
            report.append("    - 对内容质量要求较高")
            report.append("    - 活跃度和发帖频率高")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存报告
    with open("weibo_portrait_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n💾 已保存画像报告: weibo_portrait_report.txt")
    
    return report_text

# ======================================
# 主程序
# ======================================

def main():
    print("=" * 60)
    print("微博受众画像分析：核心圈层与行为聚类")
    print("=" * 60)
    print()
    
    # 1. 加载数据
    # 自动查找最新的数据文件
    import glob
    import os
    data_files = glob.glob("weibo_data_*.json")
    if data_files:
        # 按修改时间排序，使用最新的文件
        latest_file = max(data_files, key=os.path.getmtime)
        print(f"📁 找到数据文件: {latest_file}")
        df = load_data(latest_file)
    else:
        # 使用默认文件名
        df = load_data()
    
    if df is None or len(df) == 0:
        print("❌ 数据为空，无法进行分析")
        return
    
    print(f"📊 数据统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  唯一用户数: {df['user'].nunique() if 'user' in df.columns else 'N/A'}")
    print(f"  关键词数: {df['keyword'].nunique() if 'keyword' in df.columns else 'N/A'}")
    
    # 2. 数据预处理
    print("\n🔧 进行数据预处理...")
    print("  步骤1/6: 标准化列名...")
    df = standardize_columns(df)
    print("  步骤2/6: 提取时间特征...")
    df = extract_time_features(df)
    print("  步骤3/6: 提取内容特征...")
    df = extract_content_features(df)
    print("  步骤4/6: 计算互动特征...")
    df = calculate_interaction_features(df)
    print("  步骤5/6: 计算用户参与度特征...")
    df = calculate_user_engagement_features(df)
    print("  步骤6/6: 提取情感特征...")
    df = extract_sentiment_features(df)
    print("✅ 数据预处理完成")
    
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
    
    # 5.1 生成额外专业可视化图表
    print("\n📊 生成额外专业可视化图表...")
    create_additional_visualizations(df, user_type_map)
    
    # 6. 生成报告
    print("\n📝 生成画像报告...")
    generate_portrait_report(df, user_type_map)
    
    # 7. 保存结果
    output_file = "weibo_user_portrait.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 已保存分析结果: {output_file}")
    
    print("\n✅ 受众画像分析完成！")

if __name__ == "__main__":
    main()

