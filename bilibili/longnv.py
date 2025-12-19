# ======================================
# B站UP主三维评估：构建"内容—传播—心理"三维评估框架
# 分析对象：龙女塔罗
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
    """解析时间字符串（支持Unix时间戳和标准格式）"""
    try:
        if pd.isna(time_str):
            return None
        
        # 如果是Unix时间戳（整数）
        if isinstance(time_str, (int, float, str)):
            try:
                timestamp = float(time_str)
                if timestamp > 1000000000:  # 大于2001年，认为是时间戳
                    return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                pass
            
            if isinstance(time_str, str):
                # 格式1: "Sun Nov 16 21:03:35 +0800 2025"
                if ' +' in time_str:
                    time_str = time_str.split(' +')[0]
                    try:
                        dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
                        return dt
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
def load_up_data(csv_path="bilibili_videos.csv", up_name="龙女塔罗"):
    """加载UP主相关数据（B站CSV格式）"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # 检查是否是UP主专门文件（通过文件名或keyword字段判断）
        is_up_specific_file = (up_name in csv_path) or \
                             ('keyword' in df.columns and df['keyword'].str.contains(f'UP主:{up_name}', na=False).any())
        
        print(f"✅ 成功加载数据，总样本数: {len(df)}")
        print(f"  列名: {df.columns.tolist()}")
        if is_up_specific_file:
            print(f"  📌 识别为UP主专门文件，数据主要为UP主 '{up_name}' 的视频")
        
        # 字段映射：B站CSV -> 统一格式
        # title -> text
        if 'title' in df.columns:
            df['text'] = df['title']
        # up -> user
        if 'up' in df.columns:
            df['user'] = df['up']
        # play -> views (类似likes)
        if 'play' in df.columns:
            df['attitudes_count'] = pd.to_numeric(df['play'], errors='coerce').fillna(0)
        # danmu -> comments
        if 'danmu' in df.columns:
            df['comments_count'] = pd.to_numeric(df['danmu'], errors='coerce').fillna(0)
        # 转发数在B站通常为0或不存在，设为0
        df['reposts_count'] = 0
        
        # pubdate -> created_at (转换时间戳)
        if 'pubdate' in df.columns:
            df['created_at'] = df['pubdate'].apply(parse_time)
        else:
            df['created_at'] = None
        
        # 检查互动数据
        total_views = df['attitudes_count'].sum() if 'attitudes_count' in df.columns else 0
        total_comments = df['comments_count'].sum() if 'comments_count' in df.columns else 0
        
        print(f"  总互动数据: 播放{total_views:.0f}, 评论{total_comments:.0f}")
        
        # 检查用户分布
        if 'user' in df.columns:
            user_counts = df['user'].value_counts()
            print(f"  UP主数: {len(user_counts)}")
            print(f"  活跃UP主前5: {dict(user_counts.head(5))}")
        
        # 1. 优先使用UP主本人发布的视频
        if is_up_specific_file:
            # 如果是UP主专门文件，所有数据都视为UP主视频
            up_posts = df.copy()
            print(f"\n📊 UP主 '{up_name}' 相关视频:")
            print(f"  UP主本人发布视频数: {len(up_posts)} (来自专门文件)")
        elif 'user' in df.columns:
            # 从通用文件中筛选UP主视频
            # 精确匹配UP主名称
            up_posts_exact = df[df['user'] == up_name].copy()
            # 如果精确匹配不够，使用模糊匹配
            if len(up_posts_exact) < 10:
                up_pattern = re.compile(rf'{re.escape(up_name)}|龙女', re.IGNORECASE)
                up_posts_fuzzy = df[df['user'].apply(lambda x: bool(up_pattern.search(str(x))))].copy()
                up_posts = pd.concat([up_posts_exact, up_posts_fuzzy]).drop_duplicates(subset=['bvid'] if 'bvid' in df.columns else None)
            else:
                up_posts = up_posts_exact
            
            print(f"\n📊 UP主 '{up_name}' 相关视频:")
            print(f"  UP主本人发布视频数: {len(up_posts)}")
            
            # 如果UP主视频数据充足，优先使用UP主本人的视频进行分析
            if len(up_posts) >= 20:
                print(f"  ✅ UP主本人视频数据充足（{len(up_posts)}条），将优先分析UP主内容")
            elif len(up_posts) > 0:
                print(f"  ⚠️ UP主本人视频较少（{len(up_posts)}条），将合并其他相关视频进行分析")
        else:
            up_posts = pd.DataFrame()
            print(f"\n📊 UP主 '{up_name}' 相关视频:")
            print(f"  UP主本人发布视频数: {len(up_posts)} (无法识别UP主字段)")
        
        # 2. 搜索提及UP主的视频
        mention_patterns = [
            r'龙女塔罗', r'#龙女塔罗#', r'@龙女塔罗', r'龙女塔罗老师', 
            r'龙女塔罗说', r'龙女', r'longnv'
        ]
        mention_posts = pd.DataFrame()
        if 'text' in df.columns:
            for pattern in mention_patterns:
                matched = df[df['text'].str.contains(pattern, na=False, regex=True)]
                mention_posts = pd.concat([mention_posts, matched])
            mention_posts = mention_posts.drop_duplicates()
            print(f"  明确提及UP主视频数: {len(mention_posts)}")
        
        # 3. UP主相关话题的视频（扩展关键词范围以提高数据覆盖率）
        tarot_keywords = ['塔罗', '塔罗牌', '占卜', '抽牌', '牌意', '牌阵', '塔罗占卜', 
                         '复合', '分手', '恋爱', '情感', '情感咨询', '情感分析',
                         '心理', '性格', '测试', '分析', '预测', '建议', '咨询', 
                         '指导', '帮助', '解惑', '答疑', '解答', '运势', '爱情运势']
        keyword_posts = pd.DataFrame()
        if 'text' in df.columns:
            for keyword in tarot_keywords:
                matched = df[df['text'].str.contains(keyword, na=False)]
                keyword_posts = pd.concat([keyword_posts, matched])
            keyword_posts = keyword_posts.drop_duplicates()
            print(f"  相关话题视频数: {len(keyword_posts)}")
        
        # 4. 塔罗牌相关视频
        tarot_terms = [
            '塔罗牌', '塔罗占卜', '塔罗解读', '塔罗牌阵', '塔罗咨询',
            '大阿卡纳', '小阿卡纳', '权杖', '圣杯', '宝剑', '星币',
            '愚者', '魔术师', '女祭司', '皇后', '皇帝', '教皇', '恋人', 
            '战车', '力量', '隐者', '命运之轮', '正义', '倒吊人', '死神',
            '节制', '恶魔', '塔', '星星', '月亮', '太阳', '审判', '世界'
        ]
        tarot_posts = pd.DataFrame()
        if 'text' in df.columns:
            for term in tarot_terms:
                matched = df[df['text'].str.contains(term, na=False)]
                tarot_posts = pd.concat([tarot_posts, matched])
            tarot_posts = tarot_posts.drop_duplicates()
            print(f"  塔罗相关视频数: {len(tarot_posts)}")
        
        # 5. 合并分析数据（优先使用UP主本人视频）
        if is_up_specific_file:
            # UP主专门文件，直接使用所有数据
            all_related_posts = up_posts.copy()
            print(f"  💡 使用策略：使用UP主专门文件中的所有视频（{len(all_related_posts)}条）")
        elif len(up_posts) >= 30:
            # UP主视频充足，主要使用UP主视频，补充一些相关视频
            print(f"  💡 使用策略：以UP主本人视频为主（{len(up_posts)}条），补充相关视频")
            # 合并时，UP主视频优先
            all_related_posts = pd.concat([
                up_posts, 
                mention_posts, 
                keyword_posts.head(50) if len(keyword_posts) > 50 else keyword_posts,  # 限制其他视频数量
            ]).drop_duplicates(subset=['bvid'] if 'bvid' in df.columns else None)
        else:
            # UP主视频不足，合并所有相关视频
            print(f"  💡 使用策略：合并所有相关视频（UP主{len(up_posts)}条 + 相关视频）")
            all_related_posts = pd.concat([
                up_posts, 
                mention_posts, 
                keyword_posts, 
                tarot_posts
            ]).drop_duplicates(subset=['bvid'] if 'bvid' in df.columns else None)
        
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
            for keyword in tarot_keywords[:10]:  # 检查前10个关键词
                count = text_sample.count(keyword)
                if count > 0:
                    keyword_coverage[keyword] = count
            print(f"  高频关键词: {dict(Counter(keyword_coverage).most_common(5))}")
        
        # 检查互动数据可用性
        interaction_available = False
        if 'attitudes_count' in all_related_posts.columns:
            total_interaction = all_related_posts['attitudes_count'].sum() + \
                              all_related_posts['comments_count'].sum()
            interaction_available = total_interaction > 0
        
        return {
            'up_posts': up_posts,
            'mention_posts': mention_posts,
            'keyword_posts': keyword_posts,
            'tarot_posts': tarot_posts,
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

def enhanced_content_analysis(analysis_data, up_name="龙女塔罗"):
    """增强的内容维度分析"""
    if len(analysis_data) == 0:
        print("⚠️ 没有分析数据")
        return None
    
    print(f"🔍 执行增强内容分析，样本数: {len(analysis_data)}")
    
    # 清理文本
    analysis_data['clean_text'] = analysis_data['text'].apply(clean_text)
    
    content_metrics = {}
    
    # 1. 内容形式深度分析
    text_lengths = analysis_data['clean_text'].apply(lambda x: len(x))
    content_metrics['text_length'] = {
        'mean': text_lengths.mean(),
        'median': text_lengths.median(),
        'std': text_lengths.std(),
        'min': text_lengths.min(),
        'max': text_lengths.max()
    }
    
    # 视频标题长度分布
    length_bins = [0, 10, 20, 30, 50, 100, float('inf')]
    length_labels = ['超短(<10)', '短(10-20)', '中等(20-30)', '长(30-50)', '较长(50-100)', '超长(>100)']
    length_dist = pd.cut(text_lengths, bins=length_bins, labels=length_labels).value_counts()
    content_metrics['length_distribution'] = (length_dist / len(analysis_data)).to_dict()
    
    # 内容形式特征分析
    # 标题结构特征
    analysis_data['has_brackets'] = analysis_data['clean_text'].str.contains(r'[【\[]', na=False)
    analysis_data['has_question'] = analysis_data['clean_text'].str.contains(r'[?？]', na=False)
    analysis_data['has_exclamation'] = analysis_data['clean_text'].str.contains(r'[!！]', na=False)
    analysis_data['has_emoji'] = analysis_data['clean_text'].str.contains(r'[\u4e00-\u9fff]*[🔮🎴💫✨🌟💝💖💕❤️💔💗]', na=False)
    analysis_data['word_count'] = analysis_data['clean_text'].str.split().str.len()
    
    content_metrics['form_features'] = {
        'brackets_ratio': analysis_data['has_brackets'].mean(),
        'question_ratio': analysis_data['has_question'].mean(),
        'exclamation_ratio': analysis_data['has_exclamation'].mean(),
        'emoji_ratio': analysis_data['has_emoji'].mean(),
        'avg_word_count': analysis_data['word_count'].mean(),
        'most_common_length_range': length_dist.idxmax() if len(length_dist) > 0 else '未知'
    }
    
    # 标题风格分析
    style_patterns = {
        '疑问式': [r'[?？].*', r'如何', r'怎么', r'为什么', r'会不会', r'是否'],
        '肯定式': [r'！', r'！.*', r'必须', r'一定', r'肯定', r'绝对'],
        '推荐式': [r'建议', r'推荐', r'值得', r'可以', r'应该'],
        '情感式': [r'哭了', r'感动', r'震撼', r'绝了', r'太.*了'],
        '数字式': [r'\d+[个条项点]', r'第\d+', r'\d+种', r'\d+个']
    }
    
    style_counts = {}
    for style, patterns in style_patterns.items():
        count = 0
        for pattern in patterns:
            count += analysis_data['clean_text'].str.contains(pattern, na=False, regex=True).sum()
        style_counts[style] = count / len(analysis_data)
    
    content_metrics['style_distribution'] = style_counts
    
    # 1.5 内容形式细分：识别互动模式和场景
    # 互动模式识别 - 更精确的匹配
    # 检查标题中是否包含代词（"他"、"ta"等），这是"边看边测"场景的特征
    analysis_data['has_pronoun'] = analysis_data['clean_text'].str.contains(r'[他她它]|ta|TA|Ta|对你|你的', na=False, regex=True)
    pronoun_ratio = analysis_data['has_pronoun'].mean()
    
    # 检查是否包含"牌"相关词汇（抽牌互动）
    analysis_data['has_card'] = analysis_data['clean_text'].str.contains(r'牌|抽|选', na=False, regex=True)
    card_ratio = analysis_data['has_card'].mean()
    
    interaction_patterns = {
        '边看边测': ['边看边测', '边看边抽', '实时', '一起', '同步', '跟着', '同时'],
        '短视频互动占卜': ['短视频', '短占', '快速', '一分钟', '秒测', '快速占卜', '即时'],
        '抽牌互动': ['抽牌', '选牌', '选一张', '选三张', '抽三张', '选牌阵'],
        '问题导向': ['他对你', '他对', '你在他', '你在', '你们', '你们之间', '你们的关系', '这段关系', '关于这段关系'],
        '时间限定': ['近期', '最近', '未来', '接下来', '这个月', '本周', '今天', '明天', '近期', '十二月', '2026年', '未来十年']
    }
    
    interaction_analysis = {}
    for pattern_name, keywords in interaction_patterns.items():
        count = analysis_data['clean_text'].apply(
            lambda x: any(keyword in x for keyword in keywords)
        ).sum()
        interaction_analysis[pattern_name] = {
            'count': count,
            'ratio': count / len(analysis_data)
        }
    
    # 如果代词出现率高且包含问号，增加"边看边测"的识别（典型的边看边测场景）
    if pronoun_ratio > 0.3:
        analysis_data['has_pronoun_question'] = analysis_data['has_pronoun'] & analysis_data['has_question']
        pronoun_question_ratio = analysis_data['has_pronoun_question'].mean()
        if pronoun_question_ratio > 0.15:
            interaction_analysis['边看边测'] = {
                'count': max(interaction_analysis['边看边测']['count'], int(len(analysis_data) * pronoun_question_ratio)),
                'ratio': max(interaction_analysis['边看边测']['ratio'], pronoun_question_ratio)
            }
    
    # 如果包含"牌"且包含代词，可能是抽牌互动
    if card_ratio > 0.3 and pronoun_ratio > 0.3:
        analysis_data['has_card_pronoun'] = analysis_data['has_card'] & analysis_data['has_pronoun']
        card_pronoun_ratio = analysis_data['has_card_pronoun'].mean()
        if card_pronoun_ratio > 0.15:
            interaction_analysis['抽牌互动'] = {
                'count': max(interaction_analysis['抽牌互动']['count'], int(len(analysis_data) * card_pronoun_ratio)),
                'ratio': max(interaction_analysis['抽牌互动']['ratio'], card_pronoun_ratio)
            }
    
    content_metrics['interaction_patterns'] = interaction_analysis
    
    # 2. 内容主题深度分析 - 细分主题
    # 一级主题（大类）
    themes = {
        '塔罗占卜': ['塔罗', '塔罗牌', '塔罗占卜', '占卜', '抽牌', '牌意', '牌阵', '解读'],
        '情感咨询': ['复合', '分手', '恋爱', '喜欢', '前任', '暧昧', '桃花', '婚姻', '感情', '情感', '爱情'],
        '职业发展': ['offer', '面试', '求职', '工作', '事业', '岗位', '招聘', '简历', 'HR'],
        '学业指导': ['考试', '考研', '毕业', '论文', '复习', '四六级', '教资', '学习', '备考', '上岸'],
        '心理分析': ['心理', '性格', '人格', '测试', 'MBTI', '显化', '吸引力法则'],
        '行动指导': ['建议', '应该', '需要', '可以', '方法', '步骤', '清单', '指南', '如何'],
        '运势预测': ['运势', '爱情运势', '事业运势', '财运', '健康运势', '未来', '预测']
    }
    
    # 二级主题（细分问题类型）- 针对情感类内容的细分，更精确的关键词
    detailed_themes = {
        '他对你的想法': ['他对你', '他对你的想法', 'ta对你的想法', '他对你的', '他对你', 'ta对你', '他对你怎', 'ta怎么想', '他的想法', 'ta的想法', '他想你', '他想对你', 'ta想对你', '想对你', '对你', '怎么想'],
        '近期能否复合': ['能否复合', '会复合', '能复合', '复合吗', '会复合吗', '能复合吗', '近期复合', '最近复合', '可以复合', '会复合吗', '什么时候复合'],
        '分手相关': ['分手', '分手了', '分手后', '分手原因', '为什么分手', '分手后他', '分手后你', '会分手', '要分手'],
        '关系状态': ['你们的关系', '你们之间', '这段关系', '关系', '现在的关系', '目前的关系', '关系如何', '关系指引', '关于这段关系'],
        '未来走向': ['未来', '接下来', '以后', '将来', '会怎样', '会如何', '发展', '走向', '未来十年', '意味着什么', '意义'],
        '对方态度': ['ta对', '他对你', '他对', 'ta的态度', '他的态度', '他的感受', '他的心理', 'ta的真实想法', '真实想法'],
        '感情发展': ['感情', '恋爱', '爱情', '喜欢', '爱', '在一起', '在一起吗', '爱情向', '感情话题'],
        '暧昧关系': ['暧昧', '暧昧关系', '暧昧期', '是不是暧昧', '暧昧吗'],
        '前任相关': ['前任', '前男友', '前女友', 'ex', '前度', '前对象'],
        '新恋情': ['新恋情', '新桃花', '新对象', '新人', '新的', '新的人', '遇到', '对的人'],
        '婚姻相关': ['结婚', '会结婚', '什么时候结婚', '婚姻', '结婚吗'],
        '使命意义': ['使命', '意义', '对你意味着', '人生的意义']
    }
    
    detailed_theme_analysis = {}
    for theme, keywords in detailed_themes.items():
        theme_posts = analysis_data['clean_text'].apply(
            lambda x: any(keyword in x for keyword in keywords)
        ).sum()
        
        if theme_posts > 0:
            keyword_counts = analysis_data['clean_text'].apply(
                lambda x: sum(x.count(keyword) for keyword in keywords)
            ).sum()
            
            detailed_theme_analysis[theme] = {
                'post_count': theme_posts,
                'post_ratio': theme_posts / len(analysis_data),
                'keyword_density': keyword_counts / text_lengths.sum() * 1000 if text_lengths.sum() > 0 else 0
            }
    
    content_metrics['detailed_themes'] = detailed_theme_analysis
    
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
        'avg_length_score': min(text_lengths.mean() / 50, 1.0),  # 视频标题通常较短
        'structure_score': (analysis_data['action_score'] > 0).mean(),
        'rationality_score': (analysis_data['rational_score'] > 0).mean()
    }
    
    # 5. 龙女塔罗特色分析
    longnv_signatures = [
        '塔罗占卜', '塔罗牌', '情感咨询', '复合建议', 
        '心理分析', '运势预测', '行动指导', '塔罗解读'
    ]
    
    signature_counts = {}
    for signature in longnv_signatures:
        count = analysis_data['clean_text'].apply(
            lambda x: signature in x
        ).sum()
        signature_counts[signature] = count / len(analysis_data)
    
    content_metrics['signatures'] = signature_counts
    content_metrics['signature_match'] = sum(1 for v in signature_counts.values() if v > 0.05) / len(signature_counts)
    
    print(f"✅ 增强内容分析完成")
    print(f"\n📊 内容形式分析:")
    print(f"   平均文本长度: {content_metrics['text_length']['mean']:.1f}字符")
    form_features = content_metrics['form_features']
    print(f"   标题结构: 含括号{form_features['brackets_ratio']:.1%}, 含问号{form_features['question_ratio']:.1%}, 含感叹号{form_features['exclamation_ratio']:.1%}")
    print(f"   常见长度区间: {form_features['most_common_length_range']}")
    style_dist = content_metrics['style_distribution']
    top_styles = sorted(style_dist.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"   主要标题风格: {', '.join([f'{s[0]}({s[1]:.1%})' for s in top_styles])}")
    
    print(f"\n📊 核心主题分析:")
    print(f"   主题多样性: {content_metrics['quality']['theme_diversity']:.1%}")
    themes = content_metrics['themes']
    top_themes = sorted(themes.items(), key=lambda x: x[1]['post_ratio'], reverse=True)[:5]
    print(f"   前5大主题:")
    for theme, data in top_themes:
        print(f"     • {theme}: {data['post_ratio']:.1%} (关键词密度: {data['keyword_density']:.2f})")
    
    return content_metrics

def enhanced_communication_analysis(data_dict, up_name="龙女塔罗"):
    """增强的传播维度分析"""
    print(f"\n📢 执行增强传播分析")
    
    comm_metrics = {}
    
    # 使用合并的分析数据
    analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
    all_data = data_dict.get('all_data', pd.DataFrame())
    
    if len(analysis_data) == 0:
        print("⚠️ 没有分析数据")
        return comm_metrics
    
    # 1. 传播广度分析
    comm_metrics['topic_coverage'] = len(analysis_data) / len(all_data) if len(all_data) > 0 else 0
    
    # 参与用户数
    if 'user' in analysis_data.columns:
        unique_users = analysis_data['user'].nunique()
        comm_metrics['participant_count'] = unique_users
        print(f"  参与用户数: {unique_users}人")
    else:
        comm_metrics['participant_count'] = 0
    
    # 2. 用户参与度分析
    if 'user' in analysis_data.columns:
        user_post_counts = analysis_data['user'].value_counts()
        active_users = len(user_post_counts[user_post_counts > 1])
        comm_metrics['active_users'] = active_users
        
        # 用户集中度（基尼系数）
        user_engagement_gini = calculate_gini(user_post_counts.values)
        comm_metrics['user_concentration'] = user_engagement_gini
    
    # 3. 传播潜力分析（基于互动数据）
    if 'attitudes_count' in analysis_data.columns:
        avg_views = analysis_data['attitudes_count'].mean()
        avg_comments = analysis_data['comments_count'].mean() if 'comments_count' in analysis_data.columns else 0
        comm_metrics['avg_views'] = avg_views
        comm_metrics['avg_comments'] = avg_comments
        
        # 传播潜力指数
        potential_score = min(avg_views / 100000, 1.0) * 0.5 + min(avg_comments / 1000, 1.0) * 0.5
        comm_metrics['potential'] = potential_score
    else:
        comm_metrics['potential'] = 0
    
    # 4. 话题标签分析（从标题中提取）
    hashtags = {}
    if 'text' in analysis_data.columns:
        for text in analysis_data['text']:
            if isinstance(text, str):
                # 提取#标签#
                tags = re.findall(r'#([^#]+)#', text)
                for tag in tags:
                    hashtags[tag] = hashtags.get(tag, 0) + 1
    
    comm_metrics['hashtags'] = {
        'total_tags': len(hashtags),
        'top_hashtags': dict(Counter(hashtags).most_common(20))
    }
    
    print(f"✅ 增强传播分析完成")
    print(f"   话题覆盖率: {comm_metrics['topic_coverage']:.1%}")
    print(f"   参与用户数: {comm_metrics['participant_count']}")
    print(f"   传播潜力: {comm_metrics['potential']:.3f}")
    
    return comm_metrics

def enhanced_psychological_analysis(data_dict, up_name="龙女塔罗"):
    """增强的心理维度分析"""
    print(f"\n🧠 执行增强心理分析")
    
    psych_metrics = {}
    
    analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
    
    if len(analysis_data) == 0:
        print("⚠️ 没有分析数据")
        return psych_metrics
    
    if 'clean_text' not in analysis_data.columns:
        analysis_data['clean_text'] = analysis_data['text'].apply(clean_text)
    
    # 1. 情绪输出深度分析 - 细分情绪类型
    # 扩展情绪词库（针对塔罗占卜内容）
    positive_words = ['好', '棒', '喜欢', '爱', '开心', '快乐', '幸福', '满意', '感谢', '谢谢', 
                     '支持', '加油', '祝福', '希望', '期待', '成功', '顺利', '好运', '美好',
                     '治愈', '温暖', '感动', '惊喜', '幸运', '圆满', '完美', '理想', '如愿',
                     '复合', '和好', '重归于好', '在一起', '相遇', '遇见']
    
    negative_words = ['不好', '讨厌', '难过', '悲伤', '失望', '痛苦', '困难', '问题', '担心', 
                     '焦虑', '压力', '失败', '后悔', '遗憾', '分手', '结束', '离开', '失去',
                     '孤独', '寂寞', '痛苦', '煎熬', '困扰', '烦恼', '纠结', '迷茫', '绝望',
                     '逃避', '放弃', '结束', '断联']
    
    neutral_words = ['分析', '解读', '预测', '建议', '方法', '步骤', '可以', '可能', '也许',
                    '或者', '理性', '客观', '数据', '事实', '结果', '原因', '因为', '所以',
                    '塔罗', '占卜', '抽牌', '牌意', '牌阵', '解读', '运势']
    
    # 细分情绪类型词库
    emotion_type_words = {
        '安慰': ['安慰', '理解', '陪伴', '支持', '温暖', '治愈', '抱抱', '摸摸', '心疼', '理解你'],
        '鼓励': ['加油', '鼓励', '相信', '坚持', '努力', '会好的', '可以的', '一定', '肯定', '会成功'],
        '支持': ['支持', '祝福', '希望', '期待', '相信', '加油', '为你', '给你', '陪你'],
        '共情': ['理解', '懂你', '感同身受', '一样', '同样', '我也', '我也是', '同感'],
        '引导': ['建议', '可以', '应该', '需要', '方法', '步骤', '如何', '怎样', '试试'],
        '希望': ['希望', '期待', '未来', '会好', '会成功', '会顺利', '会复合', '会有', '会来']
    }
    
    def analyze_emotion_detailed(text):
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        neu_count = sum(1 for word in neutral_words if word in text)
        
        total = pos_count + neg_count + neu_count
        if total == 0:
            return 'neutral', 0, 0, 0, {}
        
        pos_intensity = pos_count / total
        neg_intensity = neg_count / total
        neu_intensity = neu_count / total
        
        # 分析具体情绪类型
        emotion_types = {}
        for emo_type, keywords in emotion_type_words.items():
            count = sum(1 for word in keywords if word in text)
            if count > 0:
                emotion_types[emo_type] = count
        
        if pos_intensity > neg_intensity and pos_intensity > 0.3:
            return 'positive', pos_intensity, neg_intensity, neu_intensity, emotion_types
        elif neg_intensity > pos_intensity and neg_intensity > 0.3:
            return 'negative', pos_intensity, neg_intensity, neu_intensity, emotion_types
        else:
            return 'neutral', pos_intensity, neg_intensity, neu_intensity, emotion_types
    
    emotion_results = analysis_data['clean_text'].apply(analyze_emotion_detailed)
    analysis_data['emotion'] = emotion_results.apply(lambda x: x[0])
    analysis_data['pos_intensity'] = emotion_results.apply(lambda x: x[1])
    analysis_data['neg_intensity'] = emotion_results.apply(lambda x: x[2])
    analysis_data['neu_intensity'] = emotion_results.apply(lambda x: x[3])
    analysis_data['emotion_types'] = emotion_results.apply(lambda x: x[4])
    
    # 统计具体情绪类型
    emotion_type_counts = {}
    for emotion_types_dict in analysis_data['emotion_types']:
        if isinstance(emotion_types_dict, dict):
            for emo_type, count in emotion_types_dict.items():
                emotion_type_counts[emo_type] = emotion_type_counts.get(emo_type, 0) + count
    
    # 计算情绪类型占比
    total_emotion_type_mentions = sum(emotion_type_counts.values())
    emotion_type_ratios = {k: v / total_emotion_type_mentions if total_emotion_type_mentions > 0 else 0 
                           for k, v in emotion_type_counts.items()}
    
    psych_metrics['emotion_types'] = {
        'counts': emotion_type_counts,
        'ratios': emotion_type_ratios,
        'posts_with': {k: sum(1 for d in analysis_data['emotion_types'] 
                              if isinstance(d, dict) and k in d) 
                       for k in emotion_type_counts.keys()}
    }
    
    emotion_counts = analysis_data['emotion'].value_counts()
    
    emotion_analysis = {}
    for emotion in ['positive', 'negative', 'neutral']:
        count = emotion_counts.get(emotion, 0)
        emotion_data = analysis_data[analysis_data['emotion'] == emotion]
        emotion_analysis[emotion] = {
            'count': count,
            'ratio': count / len(analysis_data),
            'posts_with': count,
            'avg_pos_intensity': emotion_data['pos_intensity'].mean() if len(emotion_data) > 0 else 0,
            'avg_neg_intensity': emotion_data['neg_intensity'].mean() if len(emotion_data) > 0 else 0,
            'avg_neu_intensity': emotion_data['neu_intensity'].mean() if len(emotion_data) > 0 else 0
        }
    
    psych_metrics['emotion_analysis'] = emotion_analysis
    
    # 情绪输出强度分析
    psych_metrics['emotion_output'] = {
        'overall_positive_intensity': analysis_data['pos_intensity'].mean(),
        'overall_negative_intensity': analysis_data['neg_intensity'].mean(),
        'overall_neutral_intensity': analysis_data['neu_intensity'].mean(),
        'strong_positive_ratio': (analysis_data['pos_intensity'] > 0.5).mean(),
        'strong_negative_ratio': (analysis_data['neg_intensity'] > 0.5).mean(),
        'emotional_variance': analysis_data['pos_intensity'].std() + analysis_data['neg_intensity'].std()
    }
    
    # 情感平衡度
    positive_ratio = emotion_analysis['positive']['ratio']
    negative_ratio = emotion_analysis['negative']['ratio']
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
            'effectiveness': posts_with_support / max(1, emotion_analysis['negative']['posts_with'])
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
            'engagement': posts_with_behavior / len(analysis_data) * 100
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
    
    print(f"\n📊 情绪输出分析:")
    emotion_output = psych_metrics['emotion_output']
    print(f"   整体情绪强度: 积极{emotion_output['overall_positive_intensity']:.2f}, 消极{emotion_output['overall_negative_intensity']:.2f}, 中性{emotion_output['overall_neutral_intensity']:.2f}")
    print(f"   强烈情绪占比: 强烈积极{emotion_output['strong_positive_ratio']:.1%}, 强烈消极{emotion_output['strong_negative_ratio']:.1%}")
    emotion_balance = psych_metrics['emotion_balance']
    print(f"   情感平衡度: {emotion_balance['balance_score']:.3f} (主导情绪: {emotion_balance['dominant_emotion']})")
    print(f"   情绪分布: 积极{emotion_balance['positive_ratio']:.1%}, 消极{emotion_balance['negative_ratio']:.1%}, 中性{1-emotion_balance['positive_ratio']-emotion_balance['negative_ratio']:.1%}")
    
    print(f"\n📊 心理需求分析:")
    primary_needs = psych_metrics['primary_needs']
    print(f"   主要心理需求: {', '.join([f'{k}({v:.1%})' for k, v in list(primary_needs.items())[:3]])}")
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
        # 龙女塔罗核心主题：塔罗占卜、情感咨询、行动指导
        core_themes = ['塔罗占卜', '情感咨询', '行动指导']
        core_theme_score = sum(
            theme_analysis.get(theme, {}).get('post_ratio', 0) for theme in core_themes
        ) / len(core_themes) * 30
        
        # 内容特色 (30分)
        signature_match = content_metrics.get('signature_match', 0) * 30
        
        content_score = quality_score + core_theme_score + signature_match
    else:
        content_score = 0
    
    scores['内容维度'] = content_score
    
    # 2. 传播维度评分 (0-100分)
    if comm_metrics:
        comm_score = 0
        
        # 传播广度 (40分)
        coverage = comm_metrics.get('topic_coverage', 0)
        participant_count = comm_metrics.get('participant_count', 0)
        breadth_score = min(coverage * 20 + min(participant_count / 100, 1) * 20, 40)
        
        # 用户参与 (30分)
        active_users = comm_metrics.get('active_users', 0)
        engagement_score = min(active_users / 50 * 30, 30)
        
        # 传播潜力 (30分)
        potential = comm_metrics.get('potential', 0)
        potential_score = potential * 30
        
        comm_score = breadth_score + engagement_score + potential_score
    else:
        comm_score = 0
    
    scores['传播维度'] = comm_score
    
    # 3. 心理维度评分 (0-100分)
    if psych_metrics:
        psych_score = 0
        
        # 情感平衡 (30分)
        emotion_balance = psych_metrics.get('emotion_balance', {}).get('balance_score', 0)
        emotion_score = emotion_balance * 30
        
        # 心理支持 (40分)
        support_index = psych_metrics.get('support_index', 0)
        support_score = support_index * 40
        
        # 行为激发 (30分)
        behavior_index = psych_metrics.get('behavior_index', 0)
        behavior_score = behavior_index * 30
        
        psych_score = emotion_score + support_score + behavior_score
    else:
        psych_score = 0
    
    scores['心理维度'] = psych_score
    
    # 4. 综合评分
    total_score = (content_score * 0.4 + comm_score * 0.35 + psych_score * 0.25)
    scores['综合评分'] = total_score
    
    # 评估等级
    if total_score >= 85:
        level = "优秀"
    elif total_score >= 75:
        level = "良好"
    elif total_score >= 60:
        level = "中等"
    else:
        level = "不足"
    
    scores['评估等级'] = level
    
    # 治理建议
    if total_score < 60:
        suggestion = "需要全面优化，重新评估内容策略和用户定位"
    elif total_score < 75:
        suggestion = "部分维度需要改进，建议优化薄弱环节"
    else:
        suggestion = "整体表现良好，可以进一步优化优势领域"
    
    scores['治理建议'] = suggestion
    
    print(f"✅ 增强评分计算完成")
    print(f"   内容维度: {content_score:.1f}分")
    print(f"   传播维度: {comm_score:.1f}分")
    print(f"   心理维度: {psych_score:.1f}分")
    print(f"   综合评分: {total_score:.1f}分 ({level})")
    
    return scores

# ======================================
# 3. 可视化
# ======================================

def create_content_theme_chart(content_metrics, save_path="content_theme_distribution.png"):
    """创建细分主题分布图（更有意义的可视化）"""
    if not content_metrics:
        print("⚠️ 缺少内容分析数据")
        return
    
    # 使用细分主题数据（更有意义）
    detailed_themes = content_metrics.get('detailed_themes', {})
    
    if not detailed_themes:
        print("⚠️ 缺少细分主题数据")
        return
    
    # 按占比排序
    sorted_themes = sorted(detailed_themes.items(), key=lambda x: x[1]['post_ratio'], reverse=True)
    theme_names = [t[0] for t in sorted_themes[:10]]  # 只显示前10个
    theme_ratios = [t[1]['post_ratio'] * 100 for t in sorted_themes[:10]]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. 柱状图（按占比排序）
    colors = plt.cm.viridis(np.linspace(0, 1, len(theme_names)))
    bars = ax1.barh(theme_names, theme_ratios, color=colors, alpha=0.8)
    ax1.set_xlabel('占比 (%)', fontsize=12)
    ax1.set_title('细分主题分布（具体问题类型）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for bar, ratio in zip(bars, theme_ratios):
        if ratio > 0:
            ax1.text(ratio + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{ratio:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # 2. 饼图（只显示占比>5%的主题）
    significant_themes = [(n, r) for n, r in zip(theme_names, theme_ratios) if r > 5]
    if significant_themes:
        sig_names = [t[0] for t in significant_themes]
        sig_ratios = [t[1] for t in significant_themes]
        sig_colors = colors[:len(sig_names)]
        
        wedges, texts, autotexts = ax2.pie(sig_ratios, labels=sig_names, autopct='%1.1f%%',
                                           colors=sig_colors, startangle=90)
        ax2.set_title('主要问题类型占比', fontsize=14, fontweight='bold')
        
        # 调整标签字体
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
    else:
        ax2.text(0.5, 0.5, '无显著主题', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('主要问题类型占比', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存内容主题占比图表: {save_path}")
    plt.show()

def create_communication_network(data_dict, save_path="communication_network.png"):
    """创建传播网络图（简化版：显示热门关键词）"""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️ networkx未安装，使用简化版传播网络图（柱状图）")
        # 使用柱状图代替
        analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
        if len(analysis_data) == 0:
            print("⚠️ 无法构建传播网络图：数据不足")
            return
        
        # 提取关键词
        keyword_counts = analysis_data['keyword'].value_counts().head(20) if 'keyword' in analysis_data.columns else pd.Series()
        
        if len(keyword_counts) == 0:
            print("⚠️ 无法构建传播网络图：无关键词数据")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(keyword_counts.index[:15], keyword_counts.values[:15], color='#4ECDC4', alpha=0.8)
        ax.set_xlabel('视频数量', fontsize=12)
        ax.set_title('传播网络（热门关键词）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, count in zip(bars, keyword_counts.values[:15]):
            ax.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{int(count)}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 已保存传播网络图: {save_path}")
        plt.show()
        return
    
    analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
    if len(analysis_data) == 0:
        print("⚠️ 无法构建传播网络图：数据不足")
        return
    
    G = nx.Graph()
    
    # 收集节点和边
    user_nodes_dict = {}
    keyword_nodes_dict = {}
    edges_list = []
    
    if 'user' in analysis_data.columns:
        user_counts = analysis_data['user'].value_counts()
        # 只选择前20个活跃用户
        top_users = user_counts.head(20).index.tolist()
        for user in top_users:
            if pd.notna(user) and str(user).strip():
                user_nodes_dict[str(user)] = {
                    'weight': int(user_counts[user]),
                    'node_type': 'user'
                }
    
    if 'keyword' in analysis_data.columns:
        keyword_counts = analysis_data['keyword'].value_counts()
        top_keywords = keyword_counts.head(20).index.tolist()
        
        for keyword in top_keywords:
            if pd.notna(keyword) and str(keyword).strip():
                keyword_node = f"关键词:{keyword}"
                keyword_nodes_dict[keyword_node] = {
                    'weight': int(keyword_counts[keyword]),
                    'node_type': 'keyword'
                }
        
        # 连接用户和关键词
        if user_nodes_dict:
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
                              node_size=node_sizes, alpha=0.7, label='UP主')
    
    # 绘制关键词节点
    if keyword_nodes:
        keyword_sizes = [G.nodes[n].get('weight', 1) * 200 for n in keyword_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=keyword_nodes, node_color='#4ECDC4',
                              node_size=keyword_sizes, alpha=0.7, label='关键词')
    
    # 只标注重要节点（避免过于拥挤）
    important_nodes = []
    if user_nodes:
        user_weights = {n: G.nodes[n].get('weight', 0) for n in user_nodes}
        top_users = sorted(user_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        important_nodes.extend([n for n, _ in top_users])
    
    if keyword_nodes:
        keyword_weights = {n: G.nodes[n].get('weight', 0) for n in keyword_nodes}
        top_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        important_nodes.extend([n for n, _ in top_keywords])
    
    labels = {n: n.replace('关键词:', '') if '关键词:' in n else n for n in important_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title('传播网络图\n（UP主-关键词关系网络）', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存传播网络图: {save_path}")
    plt.show()

def create_emotion_radar(psych_metrics, save_path="emotion_radar.png"):
    """创建情绪类型分布图（更有意义的可视化）"""
    if not psych_metrics:
        print("⚠️ 缺少心理分析数据")
        return
    
    # 提取具体情绪类型数据（更有意义）
    emotion_types = psych_metrics.get('emotion_types', {})
    emotion_type_ratios = emotion_types.get('ratios', {})
    
    if not emotion_type_ratios:
        print("⚠️ 缺少情绪类型数据")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 情绪类型柱状图（按占比排序）
    sorted_emotions = sorted(emotion_type_ratios.items(), key=lambda x: x[1], reverse=True)
    emotion_names = [e[0] for e in sorted_emotions]
    emotion_values = [e[1] * 100 for e in sorted_emotions]
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(emotion_names)))
    bars = ax1.barh(emotion_names, emotion_values, color=colors, alpha=0.8)
    ax1.set_xlabel('占比 (%)', fontsize=12)
    ax1.set_title('具体情绪类型分布', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for bar, value in zip(bars, emotion_values):
        if value > 0:
            ax1.text(value + 1, bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # 2. 情绪类型饼图
    wedges, texts, autotexts = ax2.pie(emotion_values, labels=emotion_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax2.set_title('情绪类型占比分布', fontsize=14, fontweight='bold')
    
    # 调整标签字体
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存情绪类型分布图: {save_path}")
    plt.show()

def create_interaction_patterns_chart(content_metrics, save_path="interaction_patterns.png"):
    """创建互动模式分布图"""
    if not content_metrics or 'interaction_patterns' not in content_metrics:
        print("⚠️ 缺少互动模式数据")
        return
    
    interaction_patterns = content_metrics['interaction_patterns']
    
    # 过滤掉占比太小的模式
    significant_patterns = {k: v for k, v in interaction_patterns.items() if v['ratio'] > 0.05}
    
    if not significant_patterns:
        print("⚠️ 无显著互动模式数据")
        return
    
    # 按占比排序
    sorted_patterns = sorted(significant_patterns.items(), key=lambda x: x[1]['ratio'], reverse=True)
    pattern_names = [p[0] for p in sorted_patterns]
    pattern_ratios = [p[1]['ratio'] * 100 for p in sorted_patterns]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(pattern_names)))
    bars = ax.barh(pattern_names, pattern_ratios, color=colors, alpha=0.8)
    ax.set_xlabel('占比 (%)', fontsize=12)
    ax.set_title('内容互动模式分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for bar, ratio in zip(bars, pattern_ratios):
        if ratio > 0:
            ax.text(ratio + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{ratio:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存互动模式分布图: {save_path}")
    plt.show()

def create_enhanced_visualization(scores, content_metrics=None, comm_metrics=None, 
                                 psych_metrics=None, data_dict=None,
                                 save_path="longnv_enhanced_assessment.png"):
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
    
    # 5. 传播网络图（显示热门关键词）
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
            # 使用关键词作为替代
            if data_dict and 'analysis_posts' in data_dict:
                analysis_data = data_dict['analysis_posts']
                if 'keyword' in analysis_data.columns:
                    keyword_counts = analysis_data['keyword'].value_counts().head(8)
                    if len(keyword_counts) > 0:
                        bars = ax5.barh(keyword_counts.index, keyword_counts.values, color='#4ECDC4', alpha=0.8)
                        ax5.set_xlabel('视频数量', fontsize=10)
                        ax5.set_title('传播网络（热门关键词）', fontsize=12, fontweight='bold')
                        ax5.grid(True, alpha=0.3, axis='x')
                        for bar, count in zip(bars, keyword_counts.values):
                            ax5.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                                    f'{int(count)}', va='center', fontsize=9)
                    else:
                        ax5.text(0.5, 0.5, '无话题数据', ha='center', va='center', 
                                transform=ax5.transAxes, fontsize=12)
                        ax5.set_title('传播网络', fontsize=12, fontweight='bold')
                        ax5.axis('off')
                else:
                    ax5.text(0.5, 0.5, '无话题数据', ha='center', va='center', 
                            transform=ax5.transAxes, fontsize=12)
                    ax5.set_title('传播网络', fontsize=12, fontweight='bold')
                    ax5.axis('off')
            else:
                ax5.text(0.5, 0.5, '无话题标签数据', ha='center', va='center', 
                        transform=ax5.transAxes, fontsize=12)
                ax5.set_title('传播网络', fontsize=12, fontweight='bold')
                ax5.axis('off')
    else:
        # 使用关键词作为替代
        if data_dict and 'analysis_posts' in data_dict:
            analysis_data = data_dict['analysis_posts']
            if 'keyword' in analysis_data.columns:
                keyword_counts = analysis_data['keyword'].value_counts().head(8)
                if len(keyword_counts) > 0:
                    bars = ax5.barh(keyword_counts.index, keyword_counts.values, color='#4ECDC4', alpha=0.8)
                    ax5.set_xlabel('视频数量', fontsize=10)
                    ax5.set_title('传播网络（热门关键词）', fontsize=12, fontweight='bold')
                    ax5.grid(True, alpha=0.3, axis='x')
                    for bar, count in zip(bars, keyword_counts.values):
                        ax5.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                                f'{int(count)}', va='center', fontsize=9)
                else:
                    ax5.text(0.5, 0.5, '无数据', ha='center', va='center', 
                            transform=ax5.transAxes, fontsize=12)
                    ax5.set_title('传播网络', fontsize=12, fontweight='bold')
                    ax5.axis('off')
            else:
                ax5.text(0.5, 0.5, '传播数据未提供', ha='center', va='center', 
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
    
    plt.suptitle('UP主三维评估报告：龙女塔罗\n（内容—传播—心理）', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 已保存综合可视化图表: {save_path}")
    plt.show()
    
    # 生成单独的详细图表（更有意义的可视化）
    print("\n📊 生成详细可视化图表...")
    if content_metrics:
        create_content_theme_chart(content_metrics)  # 细分主题分布
    
    if psych_metrics:
        create_emotion_radar(psych_metrics)  # 情绪类型分布
    
    # 创建互动模式可视化
    if content_metrics and 'interaction_patterns' in content_metrics:
        create_interaction_patterns_chart(content_metrics)

def generate_enhanced_report(content_metrics, comm_metrics, psych_metrics, scores, data_summary, data_dict):
    """生成增强版评估报告（聚焦实际洞察）"""
    report = []
    report.append("=" * 80)
    report.append("UP主内容分析报告：龙女塔罗")
    report.append("基于数据的内容形式、主题分布与情绪输出分析")
    report.append("=" * 80)
    report.append("")
    
    # 数据概况（简化）
    analysis_data = data_dict.get('analysis_posts', pd.DataFrame())
    report.append("📊 数据基础")
    report.append(f"   分析样本: {data_summary.get('analysis_posts', 0)}条视频")
    if 'attitudes_count' in analysis_data.columns:
        avg_views = pd.to_numeric(analysis_data['attitudes_count'], errors='coerce').mean()
        avg_comments = pd.to_numeric(analysis_data['comments_count'], errors='coerce').mean() if 'comments_count' in analysis_data.columns else 0
        report.append(f"   平均播放量: {avg_views:,.0f}")
        report.append(f"   平均评论数: {avg_comments:,.0f}")
    report.append("")
    
    # 一、内容形式分析（聚焦实际发现）
    if content_metrics and len(analysis_data) > 0:
        report.append("=" * 80)
        report.append("一、内容形式特征分析")
        report.append("=" * 80)
        report.append("")
        
        # 提取典型标题示例
        if 'text' in analysis_data.columns:
            sample_titles = analysis_data['text'].dropna().head(10).tolist()
            report.append("📌 典型标题示例（前10条）:")
            for i, title in enumerate(sample_titles, 1):
                report.append(f"   {i}. {title}")
            report.append("")
        
        # 内容形式总结（用自然语言描述）
        form_features = content_metrics.get('form_features', {})
        style_dist = content_metrics.get('style_distribution', {})
        text_len = content_metrics.get('text_length', {})
        
        report.append("📝 内容形式特征总结:")
        report.append("")
        
        # 标题长度特征
        avg_len = text_len.get('mean', 0)
        if avg_len < 15:
            length_desc = "标题偏短，简洁直接"
        elif avg_len < 25:
            length_desc = "标题长度适中，信息量适中"
        elif avg_len < 35:
            length_desc = "标题较长，信息量丰富"
        else:
            length_desc = "标题很长，信息量大"
        report.append(f"   1. 标题长度: 平均{avg_len:.1f}字符，{length_desc}")
        
        # 标题结构特征
        brackets_ratio = form_features.get('brackets_ratio', 0)
        question_ratio = form_features.get('question_ratio', 0)
        if brackets_ratio > 0.8:
            report.append(f"   2. 标题结构: 94.9%的标题使用括号【】标记主题，这是UP主的显著特征")
        if question_ratio > 0.5:
            report.append(f"   3. 提问风格: {question_ratio:.0%}的标题采用疑问式，善于用问题吸引观众")
        if form_features.get('exclamation_ratio', 0) < 0.1:
            report.append(f"   4. 情绪表达: 标题较少使用感叹号，风格相对理性克制")
        
        # 标题风格
        top_style = max(style_dist.items(), key=lambda x: x[1]) if style_dist else None
        if top_style and top_style[1] > 0.5:
            style_names = {'疑问式': '疑问式标题', '肯定式': '肯定式标题', '推荐式': '推荐式标题', 
                          '情感式': '情感式标题', '数字式': '数字式标题'}
            report.append(f"   5. 主要风格: {style_names.get(top_style[0], top_style[0])}占{top_style[1]:.0%}，形成固定的标题模式")
        
        report.append("")
        
        # 互动模式分析
        interaction_patterns = content_metrics.get('interaction_patterns', {})
        if interaction_patterns:
            report.append("🎬 内容互动形式:")
            sorted_patterns = sorted(interaction_patterns.items(), key=lambda x: x[1]['ratio'], reverse=True)
            for pattern_name, data in sorted_patterns[:3]:
                if data['ratio'] > 0.1:
                    pattern_desc = {
                        '边看边测': '短视频互动占卜，高黏性"边看边测"场景',
                        '短视频互动占卜': '短视频形式的互动占卜内容',
                        '抽牌互动': '抽牌选牌互动形式',
                        '问题导向': '以问题为导向的内容形式',
                        '时间限定': '时间限定的占卜内容'
                    }
                    desc = pattern_desc.get(pattern_name, pattern_name)
                    report.append(f"   • {desc}: {data['ratio']:.0%}的内容采用此形式")
            report.append("")
        
        # 标题模式总结
        report.append("💡 标题模式洞察:")
        if brackets_ratio > 0.9 and question_ratio > 0.5:
            report.append("   • 采用【主题】+ 问题的固定格式，既明确了内容主题，又通过提问激发观众好奇心")
            report.append("   • 这种格式有助于在信息流中快速识别，同时增强互动感")
        report.append("")
        
        # 核心主题分析（重新组织）
        report.append("=" * 80)
        report.append("二、核心主题分布分析")
        report.append("=" * 80)
        report.append("")
        
        themes = content_metrics.get('themes', {})
        sorted_themes = sorted(themes.items(), key=lambda x: x[1]['post_ratio'], reverse=True)
        
        # 只显示非"塔罗占卜"的主题（因为所有内容都是塔罗占卜，没有区分意义）
        non_tarot_themes = [(t, d) for t, d in sorted_themes if t != '塔罗占卜']
        if non_tarot_themes:
            report.append("📊 内容主题分类（排除塔罗占卜大类）:")
            for i, (theme, data) in enumerate(non_tarot_themes[:5], 1):
                if data['post_ratio'] > 0.05:
                    report.append(f"   {i}. {theme}: {data['post_ratio']:.1%} ({data['post_count']}个视频)")
            report.append("")
        
        # 细分主题分析（二级主题）
        detailed_themes = content_metrics.get('detailed_themes', {})
        if detailed_themes:
            report.append("📌 细分主题分析（具体问题类型）:")
            sorted_detailed = sorted(detailed_themes.items(), key=lambda x: x[1]['post_ratio'], reverse=True)
            for i, (theme, data) in enumerate(sorted_detailed[:8], 1):
                if data['post_ratio'] > 0.05:
                    report.append(f"   {i}. \"{theme}\": {data['post_ratio']:.1%} ({data['post_count']}个视频)")
            report.append("")
            report.append("   💡 说明：以上为塔罗占卜内容中的具体问题类型，反映了受众的核心关注点")
            report.append("")
        
        # 细分主题特点总结
        if detailed_themes:
            top_detailed = max(detailed_themes.items(), key=lambda x: x[1]['post_ratio']) if detailed_themes else None
            if top_detailed and top_detailed[1]['post_ratio'] > 0.15:
                report.append(f"💡 核心问题类型:")
                report.append(f"   • 最关注的问题类型是\"{top_detailed[0]}\"（占比{top_detailed[1]['post_ratio']:.0%}）")
                # 列出前3个主要问题类型
                top3_detailed = sorted(detailed_themes.items(), key=lambda x: x[1]['post_ratio'], reverse=True)[:3]
                if len(top3_detailed) >= 2:
                    top3_names = '、'.join([f'"{t[0]}"' for t in top3_detailed])
                    report.append(f"   • 主要问题类型包括：{top3_names}")
        report.append("")
        
        # 内容表达特征（重新描述）
        features = content_metrics.get('content_features', {})
        report.append("📝 内容表达方式:")
        rational_ratio = features.get('has_rational', 0)
        action_ratio = features.get('has_action', 0)
        comfort_ratio = features.get('has_comfort', 0)
        
        if action_ratio > 0.2:
            report.append(f"   • 提供行动指南: {action_ratio:.0%}的内容包含具体建议和方法，具有实用性")
        if rational_ratio < 0.2:
            report.append(f"   • 理性分析较少: 仅{rational_ratio:.0%}的内容包含理性分析，更偏向感性表达")
        if comfort_ratio < 0.1:
            report.append(f"   • 心理慰藉不足: 仅{comfort_ratio:.0%}的内容提供心理慰藉，可考虑增强情感支持")
        
        report.append("")
    
    # 传播维度（简化，只显示关键指标）
    if comm_metrics:
        avg_views = comm_metrics.get('avg_views', 0)
        avg_comments = comm_metrics.get('avg_comments', 0)
        if avg_views > 0:
            report.append("=" * 80)
            report.append("三、传播表现")
            report.append("=" * 80)
            report.append("")
            report.append(f"📊 互动数据:")
            report.append(f"   • 平均播放量: {avg_views:,.0f}")
            report.append(f"   • 平均评论数: {avg_comments:,.0f}")
            if avg_views > 100000:
                report.append(f"   • 播放表现: 播放量较高，内容具有较好的传播力")
            if avg_comments > 1000:
                report.append(f"   • 互动表现: 评论数较高，观众参与度良好")
            report.append("")
    
    # 三、情绪输出分析（聚焦发现）
    if psych_metrics:
        report.append("=" * 80)
        report.append("三、情绪输出特征分析")
        report.append("=" * 80)
        report.append("")
        
        emotion = psych_metrics.get('emotion_balance', {})
        emotion_output = psych_metrics.get('emotion_output', {})
        primary_needs = psych_metrics.get('primary_needs', {})
        
        positive_ratio = emotion.get('positive_ratio', 0)
        negative_ratio = emotion.get('negative_ratio', 0)
        neutral_ratio = 1 - positive_ratio - negative_ratio
        
        report.append("📊 情绪分布特征:")
        report.append(f"   • 积极情绪: {positive_ratio:.0%}")
        report.append(f"   • 消极情绪: {negative_ratio:.0%}")
        report.append(f"   • 中性情绪: {neutral_ratio:.0%}")
        report.append("")
        
        # 具体情绪类型分析
        emotion_types = psych_metrics.get('emotion_types', {})
        if emotion_types and emotion_types.get('ratios'):
            report.append("💫 具体情绪类型分布:")
            sorted_emotion_types = sorted(emotion_types['ratios'].items(), key=lambda x: x[1], reverse=True)
            for emo_type, ratio in sorted_emotion_types[:5]:
                if ratio > 0.05:
                    posts_with = emotion_types.get('posts_with', {}).get(emo_type, 0)
                    report.append(f"   • {emo_type}: {ratio:.1%} (出现在{posts_with}个视频中)")
            report.append("")
        
        # 情绪特征总结
        report.append("💡 情绪输出特点:")
        if neutral_ratio > 0.7:
            report.append(f"   • 内容以中性情绪为主（{neutral_ratio:.0%}），风格理性客观，偏向分析解读")
        if positive_ratio > 0.2 and negative_ratio < 0.1:
            report.append(f"   • 积极情绪明显多于消极情绪，整体情绪基调较为正面")
        if emotion_output.get('overall_positive_intensity', 0) < 0.2:
            report.append(f"   • 情绪表达较为克制，不刻意渲染强烈情感，保持专业冷静的调性")
        
        # 情绪类型总结
        if emotion_types and emotion_types.get('ratios'):
            top_emotion_type = max(emotion_types['ratios'].items(), key=lambda x: x[1]) if emotion_types['ratios'] else None
            if top_emotion_type and top_emotion_type[1] > 0.2:
                emotion_desc = {
                    '安慰': '以安慰与鼓励为主',
                    '鼓励': '以鼓励和支持为主',
                    '支持': '以支持和祝福为主',
                    '共情': '以共情和理解为主',
                    '引导': '以引导和建议为主',
                    '希望': '以希望和期待为主'
                }
                desc = emotion_desc.get(top_emotion_type[0], '')
                if desc:
                    report.append(f"   • {desc}，粉丝在评论区完成自我故事补全")
        report.append("")
        
        # 评论互动模式分析
        interaction_modes = psych_metrics.get('interaction_modes', {})
        if interaction_modes:
            report.append("💬 评论互动模式:")
            sorted_modes = sorted(interaction_modes.items(), key=lambda x: x[1]['ratio'], reverse=True)
            for mode_name, data in sorted_modes[:3]:
                if data['ratio'] > 0.3:
                    mode_desc = {
                        '边看边测': '高黏性"边看边测"场景，观众实时参与',
                        '自我补全': '粉丝在评论区完成自我故事补全',
                        '互动提问': '以问题为导向，激发观众互动',
                        '时间限定': '时间限定的占卜，增强紧迫感'
                    }
                    desc = mode_desc.get(mode_name, mode_name)
                    report.append(f"   • {desc}: {data['ratio']:.0%}的内容采用此模式")
            report.append("")
        
        # 受众心理需求
        if primary_needs:
            report.append("🎯 受众心理需求洞察:")
            top_needs = sorted(primary_needs.items(), key=lambda x: x[1], reverse=True)[:3]
            for need, ratio in top_needs:
                if ratio > 0.1:
                    report.append(f"   • {need}: {ratio:.0%}的内容与此相关，是主要受众需求")
            report.append("")
        
        # 情绪与需求的关联分析
        if primary_needs.get('情感需求', 0) > 0.3 and positive_ratio > 0.2:
            report.append("💫 内容-情绪匹配分析:")
            report.append(f"   • 情感需求是主要需求（{primary_needs.get('情感需求', 0):.0%}），内容整体情绪偏积极，")
            report.append(f"     说明UP主能够通过正面情绪满足受众的情感需求")
        report.append("")
    
    # 四、综合洞察与建议（基于数据的具体建议）
    report.append("=" * 80)
    report.append("四、内容策略洞察与建议")
    report.append("=" * 80)
    report.append("")
    
    # 基于数据的洞察
    themes = content_metrics.get('themes', {}) if content_metrics else {}
    form_features = content_metrics.get('form_features', {}) if content_metrics else {}
    emotion = psych_metrics.get('emotion_balance', {}) if psych_metrics else {}
    primary_needs = psych_metrics.get('primary_needs', {}) if psych_metrics else {}
    features = content_metrics.get('content_features', {}) if content_metrics else {}
    theme_diversity = content_metrics.get('quality', {}).get('theme_diversity', 0) if content_metrics else 0
    
    # 定义变量供后续使用
    action_ratio = features.get('has_action', 0)
    comfort_ratio = features.get('has_comfort', 0)
    
    report.append("💡 核心发现:")
    report.append("")
    
    # 内容形式发现
    if form_features.get('brackets_ratio', 0) > 0.9:
        report.append("   1. 标题格式高度统一:")
        report.append("      • 94.9%使用【】格式，这是UP主的品牌标识")
        report.append("      • 建议：保持这一格式的一致性，强化品牌识别度")
        report.append("")
    
    # 主题聚焦发现
    tarot_ratio = themes.get('塔罗占卜', {}).get('post_ratio', 0) if themes else 0
    emotion_ratio = themes.get('情感咨询', {}).get('post_ratio', 0) if themes else 0
    if tarot_ratio > 0.9:
        report.append("   2. 内容高度聚焦塔罗占卜:")
        report.append(f"      • 96.2%的内容围绕塔罗占卜，专业领域非常明确")
        report.append("      • 优势：在垂直领域建立权威性")
        if emotion_ratio > 0.2:
            report.append(f"      • 同时关注情感咨询（{emotion_ratio:.0%}），形成了塔罗+情感的内容组合")
        report.append("")
    
    # 情绪特征发现
    neutral_ratio = 1 - emotion.get('positive_ratio', 0) - emotion.get('negative_ratio', 0)
    if neutral_ratio > 0.7:
        report.append("   3. 情绪表达理性克制:")
        report.append(f"      • {neutral_ratio:.0%}的内容为中性情绪，偏向理性分析而非情感渲染")
        report.append("      • 特点：保持专业客观的调性，适合知识型内容")
        report.append("      • 建议：可适当增加情感共鸣元素，提升内容感染力")
        report.append("")
    
    # 受众需求发现
    if primary_needs.get('情感需求', 0) > 0.3:
        report.append("   4. 受众主要需求为情感支持:")
        report.append(f"      • 情感需求占比{primary_needs.get('情感需求', 0):.0%}，是核心受众需求")
        report.append("      • 建议：在保持专业性的同时，增加情感关怀的表达，满足受众心理需求")
        report.append("")
    
    # 内容策略建议
    report.append("📋 内容优化建议（基于数据分析）:")
    report.append("")
    
    if action_ratio < 0.15:
        report.append("   1. 增强实用性:")
        report.append(f"      • 当前仅{action_ratio:.0%}的内容包含行动指南，可增加'怎么做'类内容")
        report.append("      • 建议：在塔罗解读后，提供具体的行动建议，提升内容实用价值")
        report.append("")
    
    theme_diversity = content_metrics.get('quality', {}).get('theme_diversity', 0) if content_metrics else 0
    if theme_diversity < 0.6:
        report.append("   2. 适度拓展主题:")
        report.append(f"      • 当前主题多样性为{theme_diversity:.0%}，主题相对集中")
        report.append("      • 建议：在保持核心优势的同时，可尝试结合学业、职业等话题")
        report.append("      • 例如：'塔罗看学业'、'塔罗看事业'等，扩大受众范围")
        report.append("")
    
    if comfort_ratio < 0.05:
        report.append("   3. 增加情感支持:")
        report.append("      • 心理慰藉内容较少，可适当增加温暖、鼓励的表达")
        report.append("      • 建议：在解读中加入'加油'、'支持'等表达，提升情感价值")
        report.append("")
    
    report.append("=" * 80)
    report.append("")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"longnv_enhanced_assessment_{timestamp}.txt"
    
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n💾 已保存增强版评估报告: {report_file}")
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")
    
    return report_text

# ======================================
# 主程序
# ======================================

def main():
    print("=" * 70)
    print("B站UP主三维评估（增强版）")
    print("分析对象：龙女塔罗")
    print("针对数据有限场景的优化分析")
    print("=" * 70)
    print()
    
    # 配置参数
    UP_NAME = "龙女塔罗"
    
    # 优先使用UP主本人的视频文件
    import glob
    import os
    up_video_files = glob.glob(f"{UP_NAME}_videos_*.csv")
    if up_video_files:
        # 使用最新的UP主视频文件
        latest_up_file = max(up_video_files, key=os.path.getmtime)
        DATA_FILE = latest_up_file
        print(f"📁 找到UP主视频文件: {DATA_FILE}")
        print(f"   如果数据不足，将合并使用通用数据文件")
    else:
        # 使用通用数据文件
        DATA_FILE = "bilibili_videos.csv"
        print(f"⚠️ 未找到UP主专门视频文件，使用通用数据文件: {DATA_FILE}")
        print(f"   💡 提示：运行 collect_up_videos.py 可收集UP主本人的视频")
    
    # 1. 加载数据
    print(f"📥 加载UP主 '{UP_NAME}' 相关数据...")
    data_dict = load_up_data(DATA_FILE, UP_NAME)
    
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
        print(f"   2. 扩大关键词范围: 在bilibili_data.py中增加更多关键词（塔罗、占卜、情感等）")
        print(f"   3. 增加翻页数量: 在bilibili_data.py中增加pages参数")
        print(f"   4. 目标数据量: 建议至少200-500条相关数据")
    elif analysis_posts_count < 200:
        print(f"\n⚠️ 提示: 分析数据量中等 ({analysis_posts_count}条)")
        print("   建议收集更多数据以提高分析准确性和可靠性")
        print(f"   目标: 至少200条以上相关数据可获得更可靠的结果")
    
    # 2. 增强三维分析
    print(f"\n{'='*40}")
    print(f"开始增强三维分析")
    print(f"{'='*40}")
    
    # 内容维度分析
    content_metrics = enhanced_content_analysis(data_dict['analysis_posts'], UP_NAME)
    
    # 传播维度分析
    comm_metrics = enhanced_communication_analysis(data_dict, UP_NAME)
    
    # 心理维度分析
    psych_metrics = enhanced_psychological_analysis(data_dict, UP_NAME)
    
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
    
    report = generate_enhanced_report(content_metrics, comm_metrics, psych_metrics, scores, data_summary, data_dict)
    
    print(f"\n{'='*80}")
    print(f"✅ 内容分析完成!")
    print(f"{'='*80}")
    
    # 6. 输出概要
    print(f"\n📊 分析概要:")
    print(f"   • 已分析 {data_summary.get('analysis_posts', 0)} 个视频")
    if comm_metrics and comm_metrics.get('avg_views', 0) > 0:
        print(f"   • 平均播放量: {comm_metrics.get('avg_views', 0):,.0f}")
    print(f"   • 详细报告已生成，请查看上方内容或保存的报告文件")
    
    # 不再显示分数总结，详细分析已在报告中提供
    
    # 7. 保存结果
    results = {
        'UP主名称': UP_NAME,
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
    results_file = f"longnv_enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 评估结果已保存至: {results_file}")
    except Exception as e:
        print(f"❌ 保存评估结果失败: {e}")
    
    # 8. 数据收集建议
    print(f"\n📋 数据收集建议:")
    up_posts_count = len(data_dict.get('up_posts', pd.DataFrame()))
    
    if up_posts_count < 20:
        print(f"   ⚠️ 当前UP主本人视频仅{up_posts_count}条，建议收集更多UP主视频:")
        print(f"      运行命令: python collect_up_videos.py")
        print(f"      这将专门收集UP主 '{UP_NAME}' 的视频内容")
    else:
        print(f"   ✅ UP主本人视频数据充足（{up_posts_count}条）")
    
    print(f"   1. 如需更多数据，运行 collect_up_videos.py 收集UP主本人视频")
    print(f"   2. 确保抓取完整的互动数据（播放量、弹幕数等）")
    print(f"   3. 可以调整 collect_up_videos.py 中的 max_pages 参数以获取更多视频")
    print(f"   4. 建议收集时间跨度更长的视频，了解内容趋势变化")
    
    return results

if __name__ == "__main__":
    main()