# ======================================
# 博主三维评估：构建"内容—传播—心理"三维评估框架 - 修正版
# ======================================

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================================
# 辅助函数
# ======================================

def calculate_gini(x):
    """计算基尼系数"""
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return 0
    cumx = np.cumsum(x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0

def clean_text(text):
    """清理文本"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@.*?\s', '', text)
    text = re.sub(r'#.*?#', '', text)
    return text.strip()

def parse_time(time_str):
    """解析时间字符串"""
    try:
        if pd.isna(time_str):
            return None
            
        if isinstance(time_str, str):
            # 格式1: "Sun Nov 16 21:03:35 +0800 2025"
            if ' +' in time_str:
                time_str = time_str.split(' +')[0]
                try:
                    # 尝试标准格式
                    dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
                    return dt
                except:
                    try:
                        # 尝试手动解析
                        months = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                            'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                            'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        
                        parts = time_str.split()
                        if len(parts) >= 5:
                            month_str = parts[1]
                            day = int(parts[2])
                            time_part = parts[3]
                            year = int(parts[4])
                            
                            if month_str in months:
                                hour, minute, second = map(int, time_part.split(':'))
                                return datetime(year, months[month_str], day, hour, minute, second)
                    except:
                        pass
            
            # 格式2: "2025-11-16 21:03:35"
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                return dt
            except:
                pass
            
            # 格式3: "2025/11/16 21:03:35"
            try:
                dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")
                return dt
            except:
                pass
        
        return None
    except Exception as e:
        return None

# ======================================
# 1. 数据加载与预处理
# ======================================
def load_blogger_data(json_path="weibo_data_20251218_012526.json", blogger_name="陶白白"):
    """加载博主相关数据"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # 检查是否是博主专门文件（通过文件名或keyword字段判断）
        import os
        import glob
        # 检查文件名是否包含博主名称或相关关键词
        is_blogger_specific_file = (
            blogger_name in json_path or
            '陶' in json_path and '白' in json_path or  # 包含"陶"和"白"的文件
            ('keyword' in df.columns and df['keyword'].str.contains(f'博主:', na=False).any())  # keyword字段包含"博主:"
        )
        
        print(f"✅ 成功加载数据，总样本数: {len(df)}")
        print(f"  列名: {df.columns.tolist()}")
        if is_blogger_specific_file:
            print(f"  📌 识别为博主专门文件，数据主要为博主 '{blogger_name}' 的微博")
        
        # 标准化列名
        if 'reposts' in df.columns:
            df['reposts_count'] = pd.to_numeric(df['reposts'], errors='coerce').fillna(0)
        if 'comments' in df.columns:
            df['comments_count'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
        if 'likes' in df.columns:
            df['attitudes_count'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
        
        # 检查互动数据
        total_reposts = df['reposts_count'].sum() if 'reposts_count' in df.columns else 0
        total_comments = df['comments_count'].sum() if 'comments_count' in df.columns else 0
        total_likes = df['attitudes_count'].sum() if 'attitudes_count' in df.columns else 0
        
        print(f"  总互动数据: 转发{total_reposts:.0f}, "
              f"评论{total_comments:.0f}, 点赞{total_likes:.0f}")
        
        # 检查用户分布
        if 'user' in df.columns:
            user_counts = df['user'].value_counts()
            print(f"  用户数: {len(user_counts)}")
            print(f"  活跃用户前5: {dict(user_counts.head(5))}")
            
            # 调试：查找可能的博主名称变体
            if not is_blogger_specific_file:
                print(f"\n  🔍 调试：查找可能的博主名称匹配...")
                # 检查是否有包含"陶"或"白"的用户名
                possible_users = [u for u in user_counts.index if isinstance(u, str) and ('陶' in u or '白' in u or 'tao' in u.lower() or 'bai' in u.lower())]
                if possible_users:
                    print(f"  找到可能相关的用户: {possible_users[:10]}")
                else:
                    print(f"  ⚠️ 未找到包含'陶'、'白'等字符的用户名")
        
        # 1. 搜索博主本人发布的微博
        if is_blogger_specific_file:
            # 如果是博主专门文件，所有数据都视为博主微博
            blogger_posts = df.copy()
            print(f"\n📊 博主 '{blogger_name}' 相关微博:")
            print(f"  博主本人发布微博数: {len(blogger_posts)} (来自专门文件)")
        elif 'user' in df.columns:
            # 从通用文件中筛选博主微博
            # 精确匹配博主名称
            blogger_posts_exact = df[df['user'] == blogger_name].copy()
            
            # 扩展匹配模式（考虑可能的变体）
            blogger_variants = [
                blogger_name,  # 精确匹配
                '陶白白',  # 明确指定
                'Taobai',  # 拼音
                'taobai',  # 小写拼音
                'TAOBAI',  # 大写拼音
            ]
            
            # 模糊匹配（包含关键词）
            blogger_patterns = [
                re.compile(rf'{re.escape(blogger_name)}', re.IGNORECASE),
                re.compile(r'陶.*白|白.*陶', re.IGNORECASE),
                re.compile(r'tao.*bai|bai.*tao', re.IGNORECASE),
            ]
            
            blogger_posts_fuzzy = pd.DataFrame()
            for pattern in blogger_patterns:
                matched = df[df['user'].apply(lambda x: bool(pattern.search(str(x))) if pd.notna(x) else False)]
                blogger_posts_fuzzy = pd.concat([blogger_posts_fuzzy, matched])
            
            blogger_posts_fuzzy = blogger_posts_fuzzy.drop_duplicates(subset=['id'] if 'id' in df.columns else None)
            
            # 合并精确匹配和模糊匹配的结果
            blogger_posts = pd.concat([blogger_posts_exact, blogger_posts_fuzzy]).drop_duplicates(subset=['id'] if 'id' in df.columns else None)
            
            print(f"\n📊 博主 '{blogger_name}' 相关微博:")
            print(f"  博主本人发布微博数: {len(blogger_posts)} (精确匹配:{len(blogger_posts_exact)}, 模糊匹配:{len(blogger_posts_fuzzy)})")
            
            if len(blogger_posts) == 0:
                print(f"  ⚠️ 未在数据中找到博主本人的微博")
                print(f"  💡 重要提示：")
                print(f"     当前使用的是通用关键词搜索数据，不包含博主本人的微博")
                print(f"     要分析博主本人的内容，请先运行收集脚本：")
                print(f"     python collect_taobaibai_weibo.py")
                print(f"     这将专门收集博主 '{blogger_name}' 的微博并生成专门的数据文件")
            
            # 如果博主微博数据充足，优先使用博主本人的微博进行分析
            if len(blogger_posts) >= 20:
                print(f"  ✅ 博主本人微博数据充足（{len(blogger_posts)}条），将优先分析博主内容")
            elif len(blogger_posts) > 0:
                print(f"  ⚠️ 博主本人微博较少（{len(blogger_posts)}条），将合并其他相关微博进行分析")
        else:
            blogger_posts = pd.DataFrame()
            print(f"\n📊 博主 '{blogger_name}' 相关微博:")
            print(f"  博主本人发布微博数: {len(blogger_posts)} (无法识别用户字段)")
        
        # 2. 搜索提及博主的微博
        mention_patterns = [
            r'陶白白', r'#陶白白#', r'@陶白白', r'陶白白老师', 
            r'陶白白说', r'陶白白星座', r'taobaibai'
        ]
        mention_posts = pd.DataFrame()
        if 'text' in df.columns:
            for pattern in mention_patterns:
                matched = df[df['text'].str.contains(pattern, na=False, regex=True)]
                mention_posts = pd.concat([mention_posts, matched])
            mention_posts = mention_posts.drop_duplicates()
            print(f"  明确提及博主微博数: {len(mention_posts)}")
        
        # 3. 博主相关话题的微博（扩展关键词范围以提高数据覆盖率）
        blogger_keywords = ['星座运势', '星座', '运势', '水逆', 'MBTI', '塔罗', '占卜', 
                           '复合', '分手', '恋爱', '情感', '情感咨询', '情感分析',
                           '心理', '性格', '人格', '测试', '分析', '预测', '建议',
                           '咨询', '指导', '帮助', '解惑', '答疑', '解答']
        keyword_posts = pd.DataFrame()
        if 'text' in df.columns:
            for keyword in blogger_keywords:
                matched = df[df['text'].str.contains(keyword, na=False)]
                keyword_posts = pd.concat([keyword_posts, matched])
            keyword_posts = keyword_posts.drop_duplicates()
            print(f"  相关话题微博数: {len(keyword_posts)}")
        
        # 4. 星座相关微博
        zodiac_keywords = [
            '白羊座', '金牛座', '双子座', '巨蟹座', '狮子座', '处女座',
            '天秤座', '天蝎座', '射手座', '摩羯座', '水瓶座', '双鱼座',
            '白羊', '金牛', '双子', '巨蟹', '狮子', '处女',
            '天秤', '天蝎', '射手', '摩羯', '水瓶', '双鱼'
        ]
        zodiac_posts = pd.DataFrame()
        if 'text' in df.columns:
            for keyword in zodiac_keywords:
                matched = df[df['text'].str.contains(keyword, na=False)]
                zodiac_posts = pd.concat([zodiac_posts, matched])
            zodiac_posts = zodiac_posts.drop_duplicates()
            print(f"  星座相关微博数: {len(zodiac_posts)}")
        
        # 5. 合并分析数据（优先使用博主本人微博）
        if is_blogger_specific_file:
            # 博主专门文件，直接使用所有数据
            all_related_posts = blogger_posts.copy()
            print(f"  💡 使用策略：使用博主专门文件中的所有微博（{len(all_related_posts)}条）")
        elif len(blogger_posts) >= 30:
            # 博主微博充足，主要使用博主微博，补充一些相关微博
            print(f"  💡 使用策略：以博主本人微博为主（{len(blogger_posts)}条），补充相关微博")
            # 合并时，博主微博优先
            all_related_posts = pd.concat([
                blogger_posts, 
                mention_posts, 
                keyword_posts.head(100) if len(keyword_posts) > 100 else keyword_posts,  # 限制其他微博数量
            ]).drop_duplicates(subset=['id'] if 'id' in df.columns else None)
        else:
            # 博主微博不足，合并所有相关微博
            print(f"  💡 使用策略：合并所有相关微博（博主{len(blogger_posts)}条 + 相关微博）")
            all_related_posts = pd.concat([
                blogger_posts, 
                mention_posts, 
                keyword_posts, 
                zodiac_posts
            ]).drop_duplicates(subset=['id'] if 'id' in df.columns else None)
        
        print(f"\n📊 综合分析数据统计:")
        print(f"  合并去重后分析数据: {len(all_related_posts)}条")
        print(f"  数据覆盖率: {len(all_related_posts)/len(df)*100:.1f}%")
        
        # 数据量评估
        if len(all_related_posts) < 200:
            print(f"  ⚠️ 数据量较少（{len(all_related_posts)}条），建议至少200条以上获得更可靠的分析结果")
        elif len(all_related_posts) < 500:
            print(f"  ⚠️ 数据量中等（{len(all_related_posts)}条），建议收集更多数据以提高分析准确性")
        else:
            print(f"  ✅ 数据量充足（{len(all_related_posts)}条），可以进行分析")
        
        # 如果相关数据太少，使用全部数据进行分析（添加标记）
        use_all_data = False
        if len(all_related_posts) < 50:
            print(f"  💡 相关数据过少，将使用全部{len(df)}条数据进行初步分析")
            all_related_posts = df.copy()
            use_all_data = True
        
        # 检查关键词覆盖率
        if len(all_related_posts) > 0 and 'text' in all_related_posts.columns:
            text_sample = all_related_posts['text'].str.cat(sep=' ')
            keyword_coverage = {}
            for keyword in blogger_keywords[:10]:  # 检查前10个关键词
                count = text_sample.count(keyword)
                if count > 0:
                    keyword_coverage[keyword] = count
            print(f"  高频关键词: {dict(Counter(keyword_coverage).most_common(5))}")
        
        # 检查互动数据可用性
        interaction_available = False
        if 'reposts_count' in all_related_posts.columns:
            total_interaction = all_related_posts['reposts_count'].sum() + \
                              all_related_posts['comments_count'].sum() + \
                              all_related_posts['attitudes_count'].sum()
            interaction_available = total_interaction > 0
        
        return {
            'blogger_posts': blogger_posts,
            'mention_posts': mention_posts,
            'keyword_posts': keyword_posts,
            'zodiac_posts': zodiac_posts,
            'analysis_posts': all_related_posts,
            'all_data': df,
            'data_summary': {
                'total_posts': len(df),
                'analysis_posts': len(all_related_posts),
                'interaction_data_available': interaction_available
            }
        }
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ======================================
# 2. 三维评估框架
# ======================================

def enhanced_content_analysis(analysis_data, blogger_name="陶白白"):
    """增强的内容维度分析"""
    if len(analysis_data) == 0:
        print("⚠️ 没有分析数据")
        return None
    
    print(f"🔍 执行增强内容分析，样本数: {len(analysis_data)}")
    
    # 清理文本
    analysis_data['clean_text'] = analysis_data['text'].apply(clean_text)
    
    content_metrics = {}
    
    # 1. 内容形式分析
    text_lengths = analysis_data['clean_text'].apply(lambda x: len(x))
    content_metrics['text_length'] = {
        'mean': text_lengths.mean(),
        'median': text_lengths.median(),
        'std': text_lengths.std(),
        'min': text_lengths.min(),
        'max': text_lengths.max()
    }
    
    # 微博长度分布
    length_bins = [0, 50, 100, 140, 200, 500, float('inf')]
    length_labels = ['超短(<50)', '短(50-100)', '中等(100-140)', '长(140-200)', '较长(200-500)', '超长(>500)']
    length_dist = pd.cut(text_lengths, bins=length_bins, labels=length_labels).value_counts()
    content_metrics['length_distribution'] = (length_dist / len(analysis_data)).to_dict()
    
    # 2. 内容主题深度分析
    # 陶白白的核心主题
    themes = {
        '星座运势': ['星座', '运势', '水逆', '星座运势', '本周运势', '下周运势', '本月运势', '年运'],
        '情感咨询': ['复合', '分手', '恋爱', '喜欢', '前任', '暧昧', '桃花', '婚姻', '感情', '情感'],
        '职业发展': ['offer', '面试', '求职', '工作', '事业', '岗位', '招聘', '简历', 'HR'],
        '学业指导': ['考试', '考研', '毕业', '论文', '复习', '四六级', '教资', '学习', '备考', '上岸'],
        '心理分析': ['MBTI', '显化', '吸引力法则', '塔罗', '占卜', '心理', '性格', '人格'],
        '行动指导': ['建议', '应该', '需要', '可以', '方法', '步骤', '清单', '指南', '如何']
    }
    
    theme_analysis = {}
    for theme, keywords in themes.items():
        # 计算主题出现频率
        theme_posts = analysis_data['clean_text'].apply(
            lambda x: any(keyword in x for keyword in keywords)
        ).sum()
        
        # 计算主题关键词密度
        keyword_counts = analysis_data['clean_text'].apply(
            lambda x: sum(x.count(keyword) for keyword in keywords)
        ).sum()
        
        theme_analysis[theme] = {
            'post_count': theme_posts,
            'post_ratio': theme_posts / len(analysis_data),
            'keyword_density': keyword_counts / text_lengths.sum() * 1000 if text_lengths.sum() > 0 else 0
        }
    
    content_metrics['themes'] = theme_analysis
    
    # 3. 内容特征分析
    # 理性预测特征
    rational_patterns = [
        '预测', '分析', '解读', '原因', '结果', '因为', '所以', 
        '逻辑', '理性', '客观', '数据', '推测', '判断', '评估'
    ]
    
    # 行动清单特征
    action_patterns = [
        '建议', '可以', '应该', '需要', '方法', '步骤', '清单', 
        '列表', '第一', '第二', '第三', '如何做', '怎么做', '行动'
    ]
    
    # 心理慰藉特征
    comfort_patterns = [
        '安慰', '鼓励', '支持', '理解', '陪伴', '共鸣', 
        '治愈', '温暖', '希望', '加油', '祝福'
    ]
    
    def count_patterns(text, patterns):
        return sum(1 for pattern in patterns if pattern in text)
    
    analysis_data['rational_score'] = analysis_data['clean_text'].apply(
        lambda x: count_patterns(x, rational_patterns)
    )
    analysis_data['action_score'] = analysis_data['clean_text'].apply(
        lambda x: count_patterns(x, action_patterns)
    )
    analysis_data['comfort_score'] = analysis_data['clean_text'].apply(
        lambda x: count_patterns(x, comfort_patterns)
    )
    
    content_metrics['content_features'] = {
        'rational_mean': analysis_data['rational_score'].mean(),
        'action_mean': analysis_data['action_score'].mean(),
        'comfort_mean': analysis_data['comfort_score'].mean(),
        'has_rational': (analysis_data['rational_score'] > 0).mean(),
        'has_action': (analysis_data['action_score'] > 0).mean(),
        'has_comfort': (analysis_data['comfort_score'] > 0).mean()
    }
    
    # 4. 内容质量评估
    # 计算内容多样性（不同主题的覆盖）
    theme_coverage = len([t for t in theme_analysis.values() if t['post_ratio'] > 0.1])
    content_metrics['quality'] = {
        'theme_diversity': theme_coverage / len(themes),
        'avg_length_score': min(text_lengths.mean() / 140, 1.0),  # 微博140字上限
        'structure_score': (analysis_data['action_score'] > 0).mean(),
        'rationality_score': (analysis_data['rational_score'] > 0).mean()
    }
    
    # 5. 陶白白特色分析
    taobaibai_signatures = [
        '星座运势分析', '理性预测', '行动清单', '情感指导', 
        '心理分析', 'MBTI性格', '复合建议', '水逆指南'
    ]
    
    signature_counts = {}
    for signature in taobaibai_signatures:
        count = analysis_data['clean_text'].apply(
            lambda x: signature in x
        ).sum()
        signature_counts[signature] = count / len(analysis_data)
    
    content_metrics['signatures'] = signature_counts
    content_metrics['signature_match'] = sum(1 for v in signature_counts.values() if v > 0.05) / len(signature_counts)
    
    print(f"✅ 增强内容分析完成")
    print(f"   平均文本长度: {content_metrics['text_length']['mean']:.1f}字符")
    print(f"   主题多样性: {content_metrics['quality']['theme_diversity']:.1%}")
    print(f"   理性内容比例: {content_metrics['content_features']['has_rational']:.1%}")
    print(f"   行动指南比例: {content_metrics['content_features']['has_action']:.1%}")
    print(f"   陶白白特征匹配度: {content_metrics['signature_match']:.1%}")
    
    return content_metrics

def enhanced_communication_analysis(data_dict, blogger_name="陶白白"):
    """增强的传播维度分析 - 针对缺少互动数据"""
    print(f"\n📢 执行增强传播分析")
    
    comm_metrics = {}
    
    # 使用合并的分析数据
    analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
    all_data = data_dict.get('all_data', pd.DataFrame())
    
    if len(analysis_data) == 0:
        print("⚠️ 没有分析数据")
        return comm_metrics
    
    # 1. 传播广度指标
    total_posts = len(all_data)
    related_posts = len(analysis_data)
    
    comm_metrics['reach'] = {
        'total_posts': total_posts,
        'related_posts': related_posts,
        'coverage_ratio': related_posts / total_posts if total_posts > 0 else 0,
        'user_count': analysis_data['user'].nunique() if 'user' in analysis_data.columns else 0
    }
    
    # 2. 传播深度指标（基于内容分析）
    # 如果缺少互动数据，使用内容特征作为代理指标
    if 'clean_text' in analysis_data.columns:
        # 计算传播潜力（内容质量指标）
        text_lengths = analysis_data['clean_text'].apply(lambda x: len(str(x)))
        avg_length = text_lengths.mean()
        
        # 高质量内容特征
        quality_features = {
            'has_questions': analysis_data['clean_text'].str.contains('[?？]').mean(),
            'has_exclamations': analysis_data['clean_text'].str.contains('[!！]').mean(),
            'has_hashtags': analysis_data['clean_text'].str.contains('#').mean(),
            'has_mentions': analysis_data['clean_text'].str.contains('@').mean(),
            'avg_sentence_length': text_lengths.mean() / (analysis_data['clean_text'].str.count('[。！？.!?]') + 1).mean()
        }
        
        comm_metrics['content_potential'] = quality_features
        
        # 传播潜力综合评分
        engagement_potential = (
            quality_features['has_questions'] * 0.3 +
            quality_features['has_exclamations'] * 0.2 +
            quality_features['has_hashtags'] * 0.3 +
            quality_features['has_mentions'] * 0.2
        )
        comm_metrics['engagement_potential'] = engagement_potential
    else:
        comm_metrics['engagement_potential'] = 0
    
    # 3. 话题扩散分析
    if 'text' in analysis_data.columns:
        # 提取话题标签
        hashtags = []
        for text in analysis_data['text'].dropna():
            matches = re.findall(r'#([^#]+)#', str(text))
            hashtags.extend(matches)
        
        if hashtags:
            hashtag_counts = Counter(hashtags)
            top_hashtags = dict(hashtag_counts.most_common(10))
            comm_metrics['hashtags'] = {
                'total_unique': len(hashtag_counts),
                'top_hashtags': top_hashtags,
                'avg_per_post': len(hashtags) / len(analysis_data)
            }
    
    # 4. 时间分布分析
    if 'created_at' in analysis_data.columns:
        try:
            # 解析时间
            analysis_data['parsed_time'] = analysis_data['created_at'].apply(parse_time)
            time_data = analysis_data.dropna(subset=['parsed_time'])
            
            if len(time_data) > 0:
                # 按小时分布
                time_data['hour'] = time_data['parsed_time'].apply(lambda x: x.hour)
                hourly_dist = time_data['hour'].value_counts().sort_index()
                
                # 活跃时段分析
                peak_hours = hourly_dist[hourly_dist > hourly_dist.quantile(0.75)].index.tolist()
                
                comm_metrics['time_distribution'] = {
                    'total_with_time': len(time_data),
                    'hourly_distribution': hourly_dist.to_dict(),
                    'peak_hours': peak_hours,
                    'temporal_consistency': len(peak_hours) / 24 if len(peak_hours) > 0 else 0
                }
        except Exception as e:
            print(f"  时间分析出错: {e}")
    
    # 5. 用户参与度（基于用户行为）
    if 'user' in analysis_data.columns:
        user_stats = analysis_data['user'].value_counts()
        comm_metrics['user_engagement'] = {
            'total_users': len(user_stats),
            'active_users': (user_stats >= 2).sum(),  # 发帖2条以上为活跃用户
            'top_users': dict(user_stats.head(5)),
            'gini_coefficient': calculate_gini(user_stats.values) if len(user_stats) > 1 else 0
        }
    
    print(f"✅ 增强传播分析完成")
    print(f"   话题覆盖率: {comm_metrics['reach']['coverage_ratio']:.1%}")
    print(f"   参与用户数: {comm_metrics['reach']['user_count']}")
    print(f"   传播潜力: {comm_metrics.get('engagement_potential', 0):.3f}")
    
    return comm_metrics

def enhanced_psychological_analysis(data_dict, blogger_name="陶白白"):
    """增强的心理维度分析"""
    print(f"\n🧠 执行增强心理分析")
    
    analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
    
    if len(analysis_data) == 0:
        print("⚠️ 没有分析数据")
        return None
    
    # 清理文本
    analysis_data['clean_text'] = analysis_data['text'].apply(clean_text)
    
    psych_metrics = {}
    
    # 1. 情感分析
    emotion_words = {
        'positive': ['开心', '高兴', '快乐', '幸福', '幸运', '顺利', '成功', '希望', '期待', '加油',
                    '祝福', '恭喜', '感谢', '感动', '温暖', '甜蜜', '美好', '满意', '优秀', '棒'],
        'negative': ['焦虑', '压力', '紧张', '担心', '害怕', '痛苦', '难过', '伤心', '失望', '绝望',
                    '生气', '愤怒', '烦恼', '纠结', '迷茫', '困惑', '孤独', '寂寞', '疲惫', '累'],
        'neutral': ['分析', '预测', '建议', '方法', '步骤', '可以', '可能', '也许', '或者', '理性',
                   '客观', '数据', '事实', '结果', '原因', '因为', '所以', '如果', '那么', '因此']
    }
    
    emotion_counts = {cat: [] for cat in emotion_words}
    for category, words in emotion_words.items():
        counts = analysis_data['clean_text'].apply(
            lambda x: sum(1 for word in words if word in x)
        )
        emotion_counts[category] = {
            'total': counts.sum(),
            'mean': counts.mean(),
            'posts_with': (counts > 0).sum(),
            'ratio': (counts > 0).sum() / len(analysis_data)
        }
    
    psych_metrics['emotion_analysis'] = emotion_counts
    
    # 情感平衡度
    positive_ratio = emotion_counts['positive']['ratio']
    negative_ratio = emotion_counts['negative']['ratio']
    emotion_balance = 1 - abs(positive_ratio - negative_ratio) / (positive_ratio + negative_ratio + 0.001)
    
    psych_metrics['emotion_balance'] = {
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'balance_score': emotion_balance,
        'dominant_emotion': 'positive' if positive_ratio > negative_ratio else 'negative' if negative_ratio > positive_ratio else 'balanced'
    }
    
    # 2. 心理需求分析
    psychological_needs = {
        '情感需求': ['爱', '喜欢', '感情', '情感', '恋爱', '分手', '复合', '婚姻', '家庭', '亲密'],
        '认知需求': ['知道', '了解', '明白', '理解', '学习', '认知', '知识', '信息', '思考', '分析'],
        '安全需求': ['安全', '稳定', '保障', '保护', '危险', '风险', '害怕', '担心', '焦虑', '压力'],
        '归属需求': ['朋友', '社交', '群体', '社区', '归属', '认同', '接受', '拒绝', '孤独', '寂寞'],
        '成长需求': ['成长', '进步', '发展', '提升', '改变', '改善', '优化', '目标', '梦想', '理想'],
        '尊重需求': ['尊重', '尊严', '面子', '名誉', '声誉', '评价', '批评', '表扬', '认可', '否定']
    }
    
    need_analysis = {}
    for need, keywords in psychological_needs.items():
        posts_with_need = analysis_data['clean_text'].apply(
            lambda x: any(keyword in x for keyword in keywords)
        ).sum()
        
        need_analysis[need] = {
            'posts': posts_with_need,
            'ratio': posts_with_need / len(analysis_data),
            'intensity': analysis_data['clean_text'].apply(
                lambda x: sum(x.count(keyword) for keyword in keywords)
            ).sum() / len(analysis_data)
        }
    
    psych_metrics['psychological_needs'] = need_analysis
    
    # 主要心理需求
    need_ratios = {need: data['ratio'] for need, data in need_analysis.items()}
    primary_needs = sorted(need_ratios.items(), key=lambda x: x[1], reverse=True)[:3]
    psych_metrics['primary_needs'] = dict(primary_needs)
    
    # 3. 心理支持效果评估
    support_indicators = {
        'advice_given': ['建议', '可以', '应该', '需要', '方法', '步骤', '如何', '怎样'],
        'comfort_provided': ['安慰', '鼓励', '支持', '理解', '陪伴', '共鸣', '温暖', '关心'],
        'solution_offered': ['解决', '处理', '应对', '面对', '克服', '改善', '调整', '改变'],
        'hope_inspired': ['希望', '未来', '明天', '加油', '坚持', '努力', '成功', '美好']
    }
    
    support_analysis = {}
    for indicator, keywords in support_indicators.items():
        posts_with_support = analysis_data['clean_text'].apply(
            lambda x: any(keyword in x for keyword in keywords)
        ).sum()
        
        support_analysis[indicator] = {
            'posts': posts_with_support,
            'ratio': posts_with_support / len(analysis_data),
            'effectiveness': posts_with_support / max(1, emotion_counts['negative']['posts_with'])
        }
    
    psych_metrics['support_analysis'] = support_analysis
    
    # 综合心理支持指数
    support_scores = [data['ratio'] for data in support_analysis.values()]
    psych_metrics['support_index'] = np.mean(support_scores) if support_scores else 0
    
    # 4. 行为激发分析
    behavior_indicators = {
        'action_intent': ['要', '想', '打算', '计划', '准备', '决定', '尝试', '开始'],
        'goal_setting': ['目标', '计划', 'flag', '打卡', '记录', '坚持', '努力', '奋斗'],
        'progress_sharing': ['分享', '告诉', '汇报', '更新', '进步', '成果', '成绩', '收获'],
        'help_seeking': ['求助', '帮忙', '帮助', '请问', '求问', '咨询', '询问', '请教']
    }
    
    behavior_analysis = {}
    for behavior, keywords in behavior_indicators.items():
        posts_with_behavior = analysis_data['clean_text'].apply(
            lambda x: any(keyword in x for keyword in keywords)
        ).sum()
        
        behavior_analysis[behavior] = {
            'posts': posts_with_behavior,
            'ratio': posts_with_behavior / len(analysis_data),
            'engagement': posts_with_behavior / len(analysis_data) * 100  # 转换为百分比
        }
    
    psych_metrics['behavior_analysis'] = behavior_analysis
    
    # 行为激发指数
    behavior_ratios = [data['ratio'] for data in behavior_analysis.values()]
    psych_metrics['behavior_index'] = np.mean(behavior_ratios) if behavior_ratios else 0
    
    # 5. 焦虑管理分析
    anxiety_terms = ['焦虑', '压力', '紧张', '担心', '害怕', '恐慌', '不安', '忧虑']
    solution_terms = ['方法', '解决', '缓解', '减少', '应对', '处理', '调整', '改善']
    
    anxiety_posts = analysis_data['clean_text'].apply(
        lambda x: any(term in x for term in anxiety_terms)
    ).sum()
    
    solution_posts = analysis_data['clean_text'].apply(
        lambda x: any(term in x for term in solution_terms)
    ).sum()
    
    anxiety_solution_posts = analysis_data['clean_text'].apply(
        lambda x: any(at in x for at in anxiety_terms) and any(st in x for st in solution_terms)
    ).sum()
    
    psych_metrics['anxiety_management'] = {
        'anxiety_mentioned': anxiety_posts / len(analysis_data),
        'solutions_provided': solution_posts / len(analysis_data),
        'targeted_solutions': anxiety_solution_posts / max(1, anxiety_posts),
        'anxiety_coverage': anxiety_solution_posts / len(analysis_data)
    }
    
    print(f"✅ 增强心理分析完成")
    print(f"   情感平衡度: {psych_metrics['emotion_balance']['balance_score']:.3f}")
    print(f"   主要心理需求: {list(psych_metrics['primary_needs'].keys())[:2]}")
    print(f"   心理支持指数: {psych_metrics['support_index']:.3f}")
    print(f"   行为激发指数: {psych_metrics['behavior_index']:.3f}")
    
    return psych_metrics

def calculate_enhanced_scores(content_metrics, comm_metrics, psych_metrics):
    """计算增强版三维评分"""
    print(f"\n📊 计算增强版三维评分...")
    
    scores = {}
    
    # 1. 内容维度评分 (0-100分)
    if content_metrics:
        content_score = 0
        
        # 内容质量 (40分)
        quality_indicators = content_metrics.get('quality', {})
        quality_score = (
            quality_indicators.get('theme_diversity', 0) * 0.4 +
            quality_indicators.get('avg_length_score', 0) * 0.3 +
            quality_indicators.get('structure_score', 0) * 0.2 +
            quality_indicators.get('rationality_score', 0) * 0.1
        ) * 40
        
        # 主题聚焦 (30分)
        theme_analysis = content_metrics.get('themes', {})
        # 陶白白核心主题：星座运势、情感咨询、行动指导
        core_themes = ['星座运势', '情感咨询', '行动指导']
        core_theme_score = sum(
            theme_analysis.get(theme, {}).get('post_ratio', 0) 
            for theme in core_themes
        ) / len(core_themes) * 30
        
        # 特征匹配 (30分)
        signature_score = (
            content_metrics.get('signature_match', 0) * 0.5 +
            content_metrics.get('content_features', {}).get('has_rational', 0) * 0.3 +
            content_metrics.get('content_features', {}).get('has_action', 0) * 0.2
        ) * 30
        
        content_score = quality_score + core_theme_score + signature_score
        scores['内容维度'] = min(max(content_score, 0), 100)
    else:
        scores['内容维度'] = 0
    
    # 2. 传播维度评分 (0-100分)
    if comm_metrics:
        comm_score = 0
        
        # 传播广度 (40分)
        reach = comm_metrics.get('reach', {})
        coverage_score = reach.get('coverage_ratio', 0) * 40
        
        # 用户参与 (30分)
        user_engagement = comm_metrics.get('user_engagement', {})
        user_score = (
            min(user_engagement.get('active_users', 0) / max(1, user_engagement.get('total_users', 1)), 1) * 0.6 +
            (1 - min(user_engagement.get('gini_coefficient', 0), 0.8)) * 0.4  # 基尼系数越低，分布越均匀
        ) * 30
        
        # 传播潜力 (30分)
        potential_score = (
            comm_metrics.get('engagement_potential', 0) * 0.5 +
            comm_metrics.get('content_potential', {}).get('has_hashtags', 0) * 0.3 +
            comm_metrics.get('content_potential', {}).get('has_mentions', 0) * 0.2
        ) * 30
        
        comm_score = coverage_score + user_score + potential_score
        scores['传播维度'] = min(max(comm_score, 0), 100)
    else:
        scores['传播维度'] = 0
    
    # 3. 心理维度评分 (0-100分)
    if psych_metrics:
        psych_score = 0
        
        # 情感支持 (35分)
        emotion = psych_metrics.get('emotion_balance', {})
        emotion_score = (
            emotion.get('balance_score', 0) * 0.6 +
            max(emotion.get('positive_ratio', 0) - emotion.get('negative_ratio', 0), 0) * 0.4
        ) * 35
        
        # 心理需求满足 (35分)
        primary_needs = psych_metrics.get('primary_needs', {})
        need_score = sum(primary_needs.values()) / len(primary_needs) if primary_needs else 0
        need_score *= 35
        
        # 支持效果 (30分)
        support_score = (
            psych_metrics.get('support_index', 0) * 0.4 +
            psych_metrics.get('behavior_index', 0) * 0.3 +
            psych_metrics.get('anxiety_management', {}).get('targeted_solutions', 0) * 0.3
        ) * 30
        
        psych_score = emotion_score + need_score + support_score
        scores['心理维度'] = min(max(psych_score, 0), 100)
    else:
        scores['心理维度'] = 0
    
    # 4. 综合评分
    if scores:
        weight_content = 0.35
        weight_comm = 0.30  # 降低传播权重，因为缺少互动数据
        weight_psych = 0.35  # 提高心理权重
        
        total_score = (
            scores.get('内容维度', 0) * weight_content +
            scores.get('传播维度', 0) * weight_comm +
            scores.get('心理维度', 0) * weight_psych
        )
        scores['综合评分'] = min(max(total_score, 0), 100)
        
        # 评估等级
        if total_score >= 85:
            scores['评估等级'] = '优秀'
            scores['治理建议'] = '三维表现均衡优秀，可继续深化专业影响力'
        elif total_score >= 70:
            scores['评估等级'] = '良好'
            scores['治理建议'] = '整体表现良好，可在薄弱环节进行针对性优化'
        elif total_score >= 60:
            scores['评估等级'] = '合格'
            scores['治理建议'] = '基本满足需求，需系统性提升内容质量和传播效果'
        elif total_score >= 40:
            scores['评估等级'] = '待改进'
            scores['治理建议'] = '需要重点改进内容质量和用户参与度'
        else:
            scores['评估等级'] = '不足'
            scores['治理建议'] = '需要全面优化，重新评估内容策略和用户定位'
    
    print(f"✅ 增强评分计算完成")
    print(f"   内容维度: {scores.get('内容维度', 0):.1f}分")
    print(f"   传播维度: {scores.get('传播维度', 0):.1f}分")
    print(f"   心理维度: {scores.get('心理维度', 0):.1f}分")
    print(f"   综合评分: {scores.get('综合评分', 0):.1f}分 ({scores.get('评估等级', '未知')})")
    
    return scores

# ======================================
# 3. 可视化与报告
# ======================================

def generate_enhanced_report(content_metrics, comm_metrics, psych_metrics, scores, data_summary):
    """生成增强版评估报告"""
    report = []
    report.append("=" * 70)
    report.append("博主三维评估报告（增强版）")
    report.append("=" * 70)
    report.append("")
    
    # 数据概况
    report.append("📊 数据概况")
    report.append(f"   总数据量: {data_summary.get('total_posts', 0)}条微博")
    report.append(f"   分析数据: {data_summary.get('analysis_posts', 0)}条相关微博")
    report.append(f"   数据覆盖率: {data_summary.get('analysis_posts', 0)/max(1, data_summary.get('total_posts', 1))*100:.1f}%")
    report.append(f"   互动数据可用性: {'是' if data_summary.get('interaction_data_available') else '否（使用增强分析）'}")
    report.append("")
    
    # 评估结果摘要
    report.append("📈 评估结果摘要")
    report.append(f"   综合评分: {scores.get('综合评分', 0):.1f}分 ({scores.get('评估等级', '未知')})")
    report.append(f"   内容维度: {scores.get('内容维度', 0):.1f}分")
    report.append(f"   传播维度: {scores.get('传播维度', 0):.1f}分")
    report.append(f"   心理维度: {scores.get('心理维度', 0):.1f}分")
    report.append("")
    
    # 详细分析
    if content_metrics:
        report.append("📝 内容维度详细分析")
        report.append("-" * 40)
        
        # 内容形式
        text_len = content_metrics.get('text_length', {})
        report.append(f"   1. 内容形式:")
        report.append(f"      • 平均长度: {text_len.get('mean', 0):.1f}字符")
        report.append(f"      • 中位数: {text_len.get('median', 0):.1f}字符")
        
        # 长度分布
        length_dist = content_metrics.get('length_distribution', {})
        for length_type, ratio in sorted(length_dist.items()):
            if ratio > 0.05:
                report.append(f"      • {length_type}: {ratio:.1%}")
        
        # 主题分析
        themes = content_metrics.get('themes', {})
        report.append(f"   2. 核心主题:")
        for theme, data in sorted(themes.items(), key=lambda x: x[1]['post_ratio'], reverse=True):
            if data['post_ratio'] > 0.1:
                report.append(f"      • {theme}: {data['post_ratio']:.1%} (密度: {data['keyword_density']:.2f})")
        
        # 内容特征
        features = content_metrics.get('content_features', {})
        report.append(f"   3. 内容特征:")
        report.append(f"      • 理性分析: {features.get('has_rational', 0):.1%}")
        report.append(f"      • 行动指南: {features.get('has_action', 0):.1%}")
        report.append(f"      • 心理慰藉: {features.get('has_comfort', 0):.1%}")
        
        report.append("")
    
    if comm_metrics:
        report.append("📢 传播维度详细分析")
        report.append("-" * 40)
        
        # 传播广度
        reach = comm_metrics.get('reach', {})
        report.append(f"   1. 传播广度:")
        report.append(f"      • 话题覆盖率: {reach.get('coverage_ratio', 0):.1%}")
        report.append(f"      • 参与用户数: {reach.get('user_count', 0)}人")
        
        # 用户参与
        user_eng = comm_metrics.get('user_engagement', {})
        if user_eng:
            report.append(f"   2. 用户参与:")
            report.append(f"      • 活跃用户: {user_eng.get('active_users', 0)}人")
            report.append(f"      • 用户集中度: {user_eng.get('gini_coefficient', 0):.3f}")
        
        # 传播潜力
        report.append(f"   3. 传播潜力:")
        report.append(f"      • 综合潜力: {comm_metrics.get('engagement_potential', 0):.3f}")
        
        # 时间分布
        time_dist = comm_metrics.get('time_distribution', {})
        if time_dist:
            report.append(f"   4. 时间分布:")
            report.append(f"      • 活跃时段: {', '.join(map(str, time_dist.get('peak_hours', [])))}点")
        
        report.append("")
    
    if psych_metrics:
        report.append("🧠 心理维度详细分析")
        report.append("-" * 40)
        
        # 情感分析
        emotion = psych_metrics.get('emotion_balance', {})
        report.append(f"   1. 情感分析:")
        report.append(f"      • 积极情绪: {emotion.get('positive_ratio', 0):.1%}")
        report.append(f"      • 消极情绪: {emotion.get('negative_ratio', 0):.1%}")
        report.append(f"      • 情感平衡度: {emotion.get('balance_score', 0):.3f}")
        
        # 心理需求
        primary_needs = psych_metrics.get('primary_needs', {})
        report.append(f"   2. 主要心理需求:")
        for need, ratio in sorted(primary_needs.items(), key=lambda x: x[1], reverse=True)[:3]:
            report.append(f"      • {need}: {ratio:.1%}")
        
        # 心理支持
        report.append(f"   3. 心理支持效果:")
        report.append(f"      • 支持指数: {psych_metrics.get('support_index', 0):.3f}")
        report.append(f"      • 行为激发: {psych_metrics.get('behavior_index', 0):.3f}")
        
        # 焦虑管理
        anxiety = psych_metrics.get('anxiety_management', {})
        report.append(f"   4. 焦虑管理:")
        report.append(f"      • 针对性解决: {anxiety.get('targeted_solutions', 0):.1%}")
        
        report.append("")
    
    # 治理建议
    report.append("💡 治理建议与优化策略")
    report.append("-" * 40)
    report.append(f"   {scores.get('治理建议', '')}")
    report.append("")
    
    # 具体建议
    content_score = scores.get('内容维度', 0)
    comm_score = scores.get('传播维度', 0)
    psych_score = scores.get('心理维度', 0)
    
    if content_score < 70:
        report.append("   1. 内容优化建议:")
        report.append("     • 增加深度分析内容，提升专业性")
        report.append("     • 加强结构化表达，提供清晰行动指南")
        report.append("     • 丰富主题内容，覆盖更多用户需求")
    
    if comm_score < 70:
        report.append("   2. 传播优化建议:")
        report.append("     • 设计互动话题，鼓励用户参与")
        report.append("     • 优化发布时间，提高内容曝光")
        report.append("     • 建立用户社群，增强用户黏性")
    
    if psych_score < 70:
        report.append("   3. 心理优化建议:")
        report.append("     • 增强情感支持内容，提供心理慰藉")
        report.append("     • 提供实用解决方案，帮助用户应对问题")
        report.append("     • 建立信任关系，提升用户心理安全感")
    
    if all(score >= 75 for score in [content_score, comm_score, psych_score]):
        report.append("   1. 整体表现优秀，建议:")
        report.append("     • 继续保持高质量内容输出")
        report.append("     • 探索新的内容形式和传播渠道")
        report.append("     • 建立品牌体系，提升长期影响力")
    
    report.append("")
    report.append("📋 评估说明")
    report.append("-" * 40)
    report.append("   • 本评估基于相关话题微博的内容分析")
    report.append("   • 在缺少互动数据的情况下，使用增强分析方法")
    report.append("   • 评估结果可用于内容策略优化和话题治理")
    report.append("   • 建议补充完整数据以获得更准确的评估")
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存报告
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"blogger_enhanced_assessment_{timestamp}.txt"
    
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n💾 已保存增强版评估报告: {report_file}")
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")
    
    return report_text

def create_content_theme_chart(content_metrics, save_path="content_theme_distribution.png"):
    """创建内容主题占比图表"""
    if not content_metrics or 'themes' not in content_metrics:
        print("⚠️ 缺少内容主题数据")
        return
    
    themes = content_metrics['themes']
    theme_names = list(themes.keys())
    theme_ratios = [themes[theme]['post_ratio'] * 100 for theme in theme_names]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(theme_names)))
    wedges, texts, autotexts = ax1.pie(theme_ratios, labels=theme_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax1.set_title('内容主题占比分布（饼图）', fontsize=14, fontweight='bold', pad=20)
    
    # 调整标签字体
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # 2. 柱状图（按占比排序）
    sorted_indices = sorted(range(len(theme_ratios)), key=lambda i: theme_ratios[i], reverse=True)
    sorted_themes = [theme_names[i] for i in sorted_indices]
    sorted_ratios = [theme_ratios[i] for i in sorted_indices]
    
    bars = ax2.barh(sorted_themes, sorted_ratios, color=colors[sorted_indices], alpha=0.8)
    ax2.set_xlabel('占比 (%)', fontsize=12)
    ax2.set_title('内容主题占比分布（柱状图）', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, ratio) in enumerate(zip(bars, sorted_ratios)):
        if ratio > 0:
            ax2.text(ratio + 0.5, i, f'{ratio:.1f}%', 
                    va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存内容主题占比图表: {save_path}")
    plt.show()

def create_communication_network(data_dict, save_path="communication_network.png"):
    """创建传播网络图"""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️ 需要安装networkx库: pip install networkx")
        return
    
    analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
    if len(analysis_data) == 0:
        print("⚠️ 没有数据，无法生成传播网络")
        return
    
    # 创建网络图
    G = nx.Graph()
    
    # 收集所有需要添加的节点
    user_nodes_dict = {}
    keyword_nodes_dict = {}
    edges_list = []
    
    # 方法1: 基于用户的传播网络（如果用户数据可用）
    if 'user' in analysis_data.columns:
        # 统计用户发帖数
        user_counts = analysis_data['user'].value_counts()
        
        # 如果数据量太大，只选择活跃用户
        if len(user_counts) > 100:
            top_users = user_counts.head(50).index.tolist()
            selected_users = [str(u) for u in top_users if pd.notna(u) and str(u).strip()]
        else:
            selected_users = [str(u) for u in user_counts.index if pd.notna(u) and str(u).strip()]
        
        # 收集用户节点
        for user in selected_users:
            if user in user_counts.index:
                user_nodes_dict[user] = {'weight': int(user_counts[user]), 'node_type': 'user'}
    
    # 方法2: 基于关键词/主题的共现网络
    if 'text' in analysis_data.columns and 'keyword' in analysis_data.columns:
        # 提取关键词
        keywords = analysis_data['keyword'].dropna().unique().tolist()
        
        # 收集关键词节点（只选择前20个热门关键词）
        keyword_counts = analysis_data['keyword'].value_counts()
        top_keywords = keyword_counts.head(20).index.tolist()
        
        for keyword in top_keywords:
            if pd.notna(keyword) and str(keyword).strip():
                keyword_node = f"关键词:{keyword}"
                keyword_nodes_dict[keyword_node] = {
                    'weight': int(keyword_counts[keyword]),
                    'node_type': 'keyword'
                }
        
        # 如果有用户数据，连接用户和关键词
        if 'user' in analysis_data.columns and user_nodes_dict:
            for idx, row in analysis_data.iterrows():
                user = str(row.get('user', ''))
                keyword = str(row.get('keyword', ''))
                if (pd.notna(row.get('user')) and pd.notna(row.get('keyword')) and
                    user in user_nodes_dict and f"关键词:{keyword}" in keyword_nodes_dict):
                    edges_list.append((user, f"关键词:{keyword}", {'weight': 1}))
    
    # 一次性添加所有节点
    G.add_nodes_from([(node, attrs) for node, attrs in user_nodes_dict.items()])
    G.add_nodes_from([(node, attrs) for node, attrs in keyword_nodes_dict.items()])
    
    # 添加边
    if edges_list:
        G.add_edges_from(edges_list)
    
    if len(G.nodes()) == 0:
        print("⚠️ 无法构建网络图：数据不足")
        return
    
    # 绘制网络图
    plt.figure(figsize=(14, 10))
    
    # 使用spring布局
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 区分节点类型
    user_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') != 'keyword' and '关键词:' not in n]
    keyword_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'keyword' or '关键词:' in n]
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color='gray')
    
    # 绘制用户节点
    if user_nodes:
        node_sizes = [G.nodes[n].get('weight', 1) * 100 for n in user_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='#FF6B6B',
                              node_size=node_sizes, alpha=0.7, label='用户')
    
    # 绘制关键词节点
    if keyword_nodes:
        keyword_sizes = [G.nodes[n].get('weight', 1) * 200 for n in keyword_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=keyword_nodes, node_color='#4ECDC4',
                              node_size=keyword_sizes, alpha=0.7, label='关键词')
    
    # 只标注重要节点（避免过于拥挤）
    important_nodes = []
    if user_nodes:
        user_weights = [(n, G.nodes[n].get('weight', 0)) for n in user_nodes]
        important_nodes.extend([n for n, w in sorted(user_weights, key=lambda x: x[1], reverse=True)[:10]])
    if keyword_nodes:
        important_nodes.extend(keyword_nodes[:10])
    
    labels = {n: n.replace('关键词:', '') if '关键词:' in n else n[:10] 
             for n in important_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title('传播网络图\n（节点大小表示参与度，连线表示关联）', 
             fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存传播网络图: {save_path}")
    plt.show()

def create_emotion_radar(psych_metrics, save_path="emotion_radar.png"):
    """创建粉丝情绪雷达图"""
    if not psych_metrics:
        print("⚠️ 缺少心理分析数据")
        return
    
    # 提取情绪数据
    emotion_analysis = psych_metrics.get('emotion_analysis', {})
    
    # 准备雷达图数据
    categories = ['积极情绪', '消极情绪', '中性情绪']
    values = [
        emotion_analysis.get('positive', {}).get('ratio', 0) * 100,
        emotion_analysis.get('negative', {}).get('ratio', 0) * 100,
        emotion_analysis.get('neutral', {}).get('ratio', 0) * 100
    ]
    
    # 如果有心理需求数据，也可以加入
    psychological_needs = psych_metrics.get('psychological_needs', {})
    if psychological_needs:
        # 选择前3个主要需求
        need_ratios = {need: data.get('ratio', 0) * 100 
                      for need, data in psychological_needs.items()}
        top_needs = sorted(need_ratios.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 扩展雷达图维度
        categories.extend([need for need, _ in top_needs])
        values.extend([ratio for _, ratio in top_needs])
    
    # 创建雷达图
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    # 计算角度
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + values[:1]  # 闭合
    angles_plot = angles + angles[:1]
    
    # 绘制雷达图
    ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#FF6B6B', label='情绪/需求占比')
    ax.fill(angles_plot, values_plot, alpha=0.25, color='#FF6B6B')
    
    # 设置标签
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # 设置范围
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.title('粉丝情绪雷达图\n（反映用户情绪分布和心理需求）', 
             fontsize=14, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # 添加数值标注
    for angle, value, category in zip(angles, values, categories):
        if value > 0:
            ax.text(angle, value + 5, f'{value:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存情绪雷达图: {save_path}")
    plt.show()

def create_enhanced_visualization(scores, content_metrics=None, comm_metrics=None, 
                                 psych_metrics=None, data_dict=None,
                                 save_path="blogger_enhanced_assessment.png"):
    """创建增强版可视化图表（包含三维评估、主题占比、传播网络、情绪雷达）"""
    
    # 创建综合可视化
    fig = plt.figure(figsize=(20, 12))
    
    dimensions = ['内容维度', '传播维度', '心理维度']
    values = [scores.get(dim, 0) for dim in dimensions]
    total_score = scores.get('综合评分', 0)
    
    # 1. 三维评估雷达图
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles_plot = angles + angles[:1]
    
    ax1.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#4ECDC4')
    ax1.fill(angles_plot, values_plot, alpha=0.25, color='#4ECDC4')
    ax1.set_xticks(angles)
    ax1.set_xticklabels(dimensions, fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([25, 50, 75, 100])
    ax1.set_title('三维评估雷达图', fontsize=12, fontweight='bold')
    
    # 2. 内容主题占比
    ax2 = plt.subplot(2, 3, 2)
    if content_metrics and 'themes' in content_metrics:
        themes = content_metrics['themes']
        theme_names = list(themes.keys())
        theme_ratios = [themes[theme]['post_ratio'] * 100 for theme in theme_names]
        
        # 只显示占比>5%的主题
        significant_themes = [(name, ratio) for name, ratio in zip(theme_names, theme_ratios) if ratio > 5]
        if significant_themes:
            names, ratios = zip(*sorted(significant_themes, key=lambda x: x[1], reverse=True))
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            bars = ax2.barh(names, ratios, color=colors, alpha=0.8)
            ax2.set_xlabel('占比 (%)', fontsize=10)
            ax2.set_title('内容主题占比', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            for bar, ratio in zip(bars, ratios):
                ax2.text(ratio + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{ratio:.1f}%', va='center', fontsize=9, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '无显著主题数据', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('内容主题占比', fontsize=12, fontweight='bold')
            ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, '主题数据未提供', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('内容主题占比', fontsize=12, fontweight='bold')
        ax2.axis('off')
    
    # 3. 粉丝情绪雷达图
    ax3 = plt.subplot(2, 3, 3, projection='polar')
    if psych_metrics and 'emotion_analysis' in psych_metrics:
        emotion_analysis = psych_metrics['emotion_analysis']
        categories = ['积极', '消极', '中性']
        values_emotion = [
            emotion_analysis.get('positive', {}).get('ratio', 0) * 100,
            emotion_analysis.get('negative', {}).get('ratio', 0) * 100,
            emotion_analysis.get('neutral', {}).get('ratio', 0) * 100
        ]
        
        angles_emo = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_emo_plot = values_emotion + values_emotion[:1]
        angles_emo_plot = angles_emo + angles_emo[:1]
        
        ax3.plot(angles_emo_plot, values_emo_plot, 'o-', linewidth=2, color='#FF6B6B')
        ax3.fill(angles_emo_plot, values_emo_plot, alpha=0.25, color='#FF6B6B')
        ax3.set_xticks(angles_emo)
        ax3.set_xticklabels(categories, fontsize=10)
        max_val = max(values_emotion) * 1.2 if max(values_emotion) > 0 else 100
        ax3.set_ylim(0, max_val)
        ax3.set_yticks([0, 25, 50, 75, 100])
        ax3.set_title('粉丝情绪雷达图', fontsize=12, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, '情绪数据未提供', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('粉丝情绪雷达图', fontsize=12, fontweight='bold')
        ax3.axis('off')
    
    # 4. 综合评分仪表盘
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    ax4.axis('off')
    
    # 绘制背景圆环
    circle = plt.Circle((0, 0), 1.0, color='lightgray', alpha=0.3)
    ax4.add_patch(circle)
    
    # 绘制评分弧
    score_angle = total_score / 100 * 180
    if total_score < 60:
        color = 'red'
    elif total_score < 75:
        color = 'orange'
    elif total_score < 85:
        color = 'yellowgreen'
    else:
        color = 'green'
    
    ax4.plot([0, 0.8 * np.sin(np.deg2rad(score_angle))], 
             [0, 0.8 * np.cos(np.deg2rad(score_angle))], 
             color=color, linewidth=4)
    
    ax4.text(0, 0, f'{total_score:.1f}', ha='center', va='center', 
             fontsize=24, fontweight='bold', color=color)
    ax4.text(0, -0.3, scores.get('评估等级', '未知'), ha='center', va='center',
             fontsize=14, fontweight='bold', color=color)
    ax4.text(0, -0.5, '综合评分', ha='center', va='center',
             fontsize=10, color='gray')
    
    # 5. 传播网络图（显示热门话题标签）
    ax5 = plt.subplot(2, 3, 5)
    if comm_metrics and 'hashtags' in comm_metrics:
        hashtags_data = comm_metrics['hashtags']
        top_hashtags = hashtags_data.get('top_hashtags', {})
        if top_hashtags:
            tags = list(top_hashtags.keys())[:8]
            counts = list(top_hashtags.values())[:8]
            colors_network = plt.cm.viridis(np.linspace(0, 1, len(tags)))
            bars = ax5.barh(tags, counts, color=colors_network, alpha=0.8)
            ax5.set_xlabel('使用次数', fontsize=10)
            ax5.set_title('传播网络（热门话题）', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            for bar, count in zip(bars, counts):
                ax5.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{int(count)}', va='center', fontsize=9)
        else:
            ax5.text(0.5, 0.5, '无话题标签数据', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('传播网络', fontsize=12, fontweight='bold')
            ax5.axis('off')
    else:
        ax5.text(0.5, 0.5, '传播数据未提供', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('传播网络', fontsize=12, fontweight='bold')
        ax5.axis('off')
    
    # 6. 建议区域
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    suggestion = scores.get('治理建议', '暂无具体建议')
    strengths = []
    weaknesses = []
    
    for dim, score in zip(dimensions, values):
        if score >= 75:
            strengths.append(f"{dim} ({score:.1f}分)")
        elif score < 60:
            weaknesses.append(f"{dim} ({score:.1f}分)")
    
    suggestion_text = f"💡 治理建议:\n\n{suggestion}\n\n"
    
    if strengths:
        suggestion_text += f"✅ 优势维度:\n" + "\n".join([f"  • {s}" for s in strengths]) + "\n\n"
    
    if weaknesses:
        suggestion_text += f"⚠️ 待改进维度:\n" + "\n".join([f"  • {w}" for w in weaknesses])
    else:
        suggestion_text += f"✅ 各维度表现均衡，无明显短板"
    
    suggestion_text += f"\n\n📊 综合评分: {total_score:.1f}分 ({scores.get('评估等级', '未知')})"
    suggestion_text += f"\n🔍 评估基于增强分析方法"
    
    ax6.text(0.05, 0.95, suggestion_text, fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
             transform=ax6.transAxes)
    
    plt.suptitle('博主三维评估报告\n（内容—传播—心理）', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存综合可视化图表: {save_path}")
    plt.show()
    
    # 生成单独的详细图表
    print("\n📊 生成详细可视化图表...")
    if content_metrics:
        create_content_theme_chart(content_metrics)
    
    if data_dict:
        create_communication_network(data_dict)
    
    if psych_metrics:
        create_emotion_radar(psych_metrics)

# ======================================
# 主程序
# ======================================

def main():
    print("=" * 70)
    print("博主三维评估（增强版）")
    print("针对数据有限场景的优化分析")
    print("=" * 70)
    print()
    
    # 配置参数
    BLOGGER_NAME = "陶白白"
    
    # 优先使用博主本人的微博文件（支持模糊匹配）
    import glob
    import os
    # 先尝试精确匹配
    blogger_weibo_files = glob.glob(f"{BLOGGER_NAME}_weibo_*.json")
    # 如果没找到，尝试模糊匹配（包含博主名称的所有weibo文件）
    if not blogger_weibo_files:
        all_weibo_files = glob.glob("*_weibo_*.json")
        blogger_weibo_files = [f for f in all_weibo_files if BLOGGER_NAME in f]
    # 如果还是没找到，尝试更宽松的匹配（包含"陶"和"白"的文件，适用于"陶白白Sensei"）
    if not blogger_weibo_files and "陶" in BLOGGER_NAME and "白" in BLOGGER_NAME:
        all_weibo_files = glob.glob("*_weibo_*.json")
        blogger_weibo_files = [f for f in all_weibo_files if "陶" in f and "白" in f and "_weibo_" in f]
    
    if blogger_weibo_files:
        # 使用最新的博主微博文件
        latest_blogger_file = max(blogger_weibo_files, key=os.path.getmtime)
        DATA_FILE = latest_blogger_file
        print(f"📁 找到博主微博文件: {DATA_FILE}")
        print(f"   如果数据不足，将合并使用通用数据文件")
    else:
        # 使用通用数据文件
        DATA_FILE = "weibo_data_20251218_012526.json"
        print(f"⚠️ 未找到博主专门微博文件，使用通用数据文件: {DATA_FILE}")
        print(f"   💡 提示：运行 collect_taobaibai_weibo.py 可收集博主本人的微博")
    
    # 1. 加载数据
    print(f"📥 加载博主 '{BLOGGER_NAME}' 相关数据...")
    data_dict = load_blogger_data(DATA_FILE, BLOGGER_NAME)
    
    if data_dict is None:
        print("❌ 数据加载失败，无法进行评估")
        return
    
    data_summary = data_dict.get('data_summary', {})
    print(f"\n📊 数据概况:")
    print(f"   总数据量: {data_summary.get('total_posts', 0)}条")
    print(f"   分析数据: {data_summary.get('analysis_posts', 0)}条")
    print(f"   互动数据可用: {data_summary.get('interaction_data_available', False)}")
    
    analysis_posts_count = data_summary.get('analysis_posts', 0)
    total_posts_count = data_summary.get('total_posts', 0)
    
    if analysis_posts_count < 50:
        print(f"\n⚠️ 警告: 分析数据较少 ({analysis_posts_count}条)")
        print("   评估结果仅供参考，建议收集更多数据")
        print(f"\n💡 数据收集建议:")
        print(f"   1. 当前数据集: 总数据{total_posts_count}条，相关数据{analysis_posts_count}条")
        print(f"   2. 扩大关键词范围: 在weibo_data.py中增加更多关键词（星座、情感、心理等）")
        print(f"   3. 增加翻页数量: 在weibo_data.py中增加MAX_PAGES参数（当前可能为20）")
        print(f"   4. 目标数据量: 建议至少1000-2000条总数据，500条以上相关数据")
    elif analysis_posts_count < 200:
        print(f"\n⚠️ 提示: 分析数据量中等 ({analysis_posts_count}条)")
        print("   建议收集更多数据以提高分析准确性和可靠性")
        print(f"   目标: 至少500条以上相关数据可获得更可靠的结果")
    
    # 2. 增强三维分析
    print(f"\n{'='*40}")
    print(f"开始增强三维分析")
    print(f"{'='*40}")
    
    # 内容维度分析
    content_metrics = enhanced_content_analysis(data_dict['analysis_posts'], BLOGGER_NAME)
    
    # 传播维度分析
    comm_metrics = enhanced_communication_analysis(data_dict, BLOGGER_NAME)
    
    # 心理维度分析
    psych_metrics = enhanced_psychological_analysis(data_dict, BLOGGER_NAME)
    
    # 3. 计算增强评分
    print(f"\n{'='*40}")
    print(f"计算增强版评分")
    print(f"{'='*40}")
    
    scores = calculate_enhanced_scores(content_metrics, comm_metrics, psych_metrics)
    
    # 4. 可视化
    print(f"\n{'='*40}")
    print(f"生成可视化图表")
    print(f"{'='*40}")
    
    create_enhanced_visualization(scores, content_metrics, comm_metrics, 
                                 psych_metrics, data_dict)
    
    # 5. 生成详细报告
    print(f"\n{'='*40}")
    print(f"生成评估报告")
    print(f"{'='*40}")
    
    report = generate_enhanced_report(content_metrics, comm_metrics, psych_metrics, scores, data_summary)
    
    print(f"\n{'='*70}")
    print(f"✅ 博主三维增强评估完成!")
    print(f"{'='*70}")
    
    # 6. 输出关键发现
    print(f"\n🔍 关键发现总结:")
    print(f"   1. 综合评分: {scores.get('综合评分', 0):.1f}分 ({scores.get('评估等级', '未知')})")
    print(f"   2. 三维表现:")
    print(f"      • 内容维度: {scores.get('内容维度', 0):.1f}分")
    print(f"      • 传播维度: {scores.get('传播维度', 0):.1f}分")
    print(f"      • 心理维度: {scores.get('心理维度', 0):.1f}分")
    print(f"   3. 主要优势: ", end="")
    
    strengths = []
    if scores.get('内容维度', 0) >= 70:
        strengths.append("内容专业性")
    if scores.get('传播维度', 0) >= 70:
        strengths.append("传播覆盖")
    if scores.get('心理维度', 0) >= 70:
        strengths.append("心理支持")
    
    if strengths:
        print(", ".join(strengths))
    else:
        print("无明显突出优势")
    
    print(f"   4. 改进重点: ", end="")
    weaknesses = []
    if scores.get('内容维度', 0) < 60:
        weaknesses.append("内容质量")
    if scores.get('传播维度', 0) < 60:
        weaknesses.append("用户参与")
    if scores.get('心理维度', 0) < 60:
        weaknesses.append("心理支持")
    
    if weaknesses:
        print(", ".join(weaknesses))
    else:
        print("暂无明确短板")
    
    # 7. 保存结果
    results = {
        '博主名称': BLOGGER_NAME,
        '评估时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '评估版本': '增强版（针对有限数据）',
        '数据概况': data_summary,
        '内容维度得分': scores.get('内容维度', 0),
        '传播维度得分': scores.get('传播维度', 0),
        '心理维度得分': scores.get('心理维度', 0),
        '综合评分': scores.get('综合评分', 0),
        '评估等级': scores.get('评估等级', '未知'),
        '治理建议': scores.get('治理建议', ''),
        '内容维度详情': content_metrics,
        '传播维度详情': comm_metrics,
        '心理维度详情': psych_metrics
    }
    
    import json
    results_file = f"blogger_enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 评估结果已保存至: {results_file}")
    except Exception as e:
        print(f"❌ 保存评估结果失败: {e}")
    
    # 8. 数据收集建议
    print(f"\n📋 数据收集建议:")
    blogger_posts_count = len(data_dict.get('blogger_posts', pd.DataFrame()))
    
    if blogger_posts_count < 20:
        print(f"   ⚠️ 当前博主本人微博仅{blogger_posts_count}条，建议收集更多博主微博:")
        print(f"      运行命令: python collect_taobaibai_weibo.py")
        print(f"      这将专门收集博主 '{BLOGGER_NAME}' 的微博内容")
    else:
        print(f"   ✅ 博主本人微博数据充足（{blogger_posts_count}条）")
    
    print(f"   1. 如需更多数据，运行 collect_taobaibai_weibo.py 收集博主本人微博")
    print(f"   2. 确保抓取完整的互动数据（转发、评论、点赞）")
    print(f"   3. 可以调整 collect_taobaibai_weibo.py 中的 max_pages 参数以获取更多微博")
    print(f"   4. 建议收集时间跨度更长的微博，了解内容趋势变化")
    
    return results

if __name__ == "__main__":
    main()