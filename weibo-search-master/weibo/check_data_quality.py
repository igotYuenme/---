"""检查数据质量并给出建议"""
import json
import pandas as pd

# 加载数据
with open('weibo_data_20251218_012526.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("=" * 70)
print("数据质量分析报告")
print("=" * 70)
print(f"\n📊 基础统计:")
print(f"   总记录数: {len(df)} 条")
print(f"   用户数: {df['user'].nunique() if 'user' in df.columns else 0}")
print(f"   关键词数: {df['keyword'].nunique() if 'keyword' in df.columns else 0}")

# 关键词分布
if 'keyword' in df.columns:
    keyword_counts = df['keyword'].value_counts()
    print(f"\n📌 关键词分布（前10）:")
    for kw, count in keyword_counts.head(10).items():
        print(f"   {kw}: {count}条 ({count/len(df)*100:.1f}%)")

# 用户分布
if 'user' in df.columns:
    user_counts = df['user'].value_counts()
    print(f"\n👥 用户分布:")
    print(f"   平均每个用户发帖数: {len(df) / len(user_counts):.1f}")
    print(f"   发帖数≥3的用户: {(user_counts >= 3).sum()}个")
    print(f"   只发1条的用户: {(user_counts == 1).sum()}个")

# 互动数据统计
if 'reposts' in df.columns and 'comments' in df.columns and 'likes' in df.columns:
    df['reposts'] = pd.to_numeric(df['reposts'], errors='coerce').fillna(0)
    df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
    
    print(f"\n💬 互动数据:")
    print(f"   平均转发: {df['reposts'].mean():.1f}")
    print(f"   平均评论: {df['comments'].mean():.1f}")
    print(f"   平均点赞: {df['likes'].mean():.1f}")
    print(f"   有互动的微博: {(df['reposts'] + df['comments'] + df['likes'] > 0).sum()}条 ({(df['reposts'] + df['comments'] + df['likes'] > 0).sum()/len(df)*100:.1f}%)")

# 数据质量评估
print(f"\n📈 数据质量评估:")
quality_issues = []

if len(df) < 500:
    quality_issues.append(f"⚠️ 数据量较少（{len(df)}条），建议至少500条以上")
    
if df['user'].nunique() < 100:
    quality_issues.append(f"⚠️ 用户数较少（{df['user'].nunique()}个），建议至少100个以上")
    
# 检查关键词分布
if 'keyword' in df.columns:
    keyword_counts = df['keyword'].value_counts()
    top_5_ratio = keyword_counts.head(5).sum() / len(df)
    if top_5_ratio > 0.7:
        quality_issues.append(f"⚠️ 关键词分布不均（前5个关键词占{top_5_ratio*100:.1f}%）")

if quality_issues:
    print("   发现以下问题:")
    for issue in quality_issues:
        print(f"   {issue}")
else:
    print("   ✅ 数据质量良好")

# 改进建议
print(f"\n💡 改进建议:")
print(f"   1. 扩大数据收集:")
print(f"      • 当前{len(df)}条，建议至少1000-2000条")
print(f"      • 可以通过增加关键词、增加翻页数来获取更多数据")
print(f"   2. 扩大关键词范围:")
print(f"      • 当前{df['keyword'].nunique() if 'keyword' in df.columns else 0}个关键词")
print(f"      • 建议添加更多相关关键词（如：情感咨询、心理分析、MBTI等）")
print(f"   3. 优化分析策略:")
if len(df) < 300:
    print(f"      • 当前数据量较少，分析结果仅供参考")
    print(f"      • 可以放宽筛选条件，使用所有{len(df)}条数据进行初步分析")
else:
    print(f"      • 当前数据量可以进行分析，但样本量越大结果越可靠")

print(f"\n" + "=" * 70)

