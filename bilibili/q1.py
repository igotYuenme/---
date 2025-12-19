# ======================================
# Q1: Decision Scenarios & Other Subscenes Esoteric Content Analysis
# 输入：bilibili_videos.csv
# 输出：
#   1. bilibili_videos_with_scene_subscene.csv
#   2. scene_ratio.csv
#   3. other_subscene_ratio.csv
#   4. scene_ratio_bar_en.png / scene_ratio_pie_en.png
#   5. other_subscene_bar_en.png / other_subscene_pie_en.png
# ======================================

import pandas as pd
import jieba
import matplotlib.pyplot as plt

# -------------------------
# Scene English Mapping
# -------------------------
scene_en_map = {
    "情感": "Emotional",
    "学业": "Academic",
    "职业": "Career",
    "日常": "Daily Life",
    "人格/自我": "Self-Identity",
    "其他": "Other"
}

# -------------------------
# 1. 读取原始数据
# -------------------------
print("📥 读取数据中...")
df = pd.read_csv("bilibili_videos.csv")
print(f"✅ 原始视频数量：{len(df)}")

# -------------------------
# 2. 定义场景关键词映射
# -------------------------
scene_map = {
    "情感": ["抽牌建议", "分手 建议"],
    "学业": ["考前 建议", "考试 运势"],
    "职业": ["面试 建议"],
    "日常": ["运势", "水逆"],
    "人格/自我": ["MBTI"],
}

def infer_scene(keyword):
    for scene, kws in scene_map.items():
        if keyword in kws:
            return scene
    return "其他"

# -------------------------
# 3. 场景标注
# -------------------------
df["scene"] = df["keyword"].apply(infer_scene)
df.to_csv("bilibili_videos_with_scene.csv", index=False, encoding="utf-8-sig")
print("💾 已保存 bilibili_videos_with_scene.csv")

# 打印场景标注统计（初步）
print(f"\n📊 场景标注完成，总视频数: {len(df)}")
scene_preview = df["scene"].value_counts()
print(f"   初步场景分布: {dict(scene_preview)}")

# -------------------------
# 4. 场景统计
# -------------------------
scene_stat = df.groupby("scene").size().reset_index(name="count")
scene_stat["ratio"] = scene_stat["count"] / scene_stat["count"].sum()
scene_stat["scene_en"] = scene_stat["scene"].map(scene_en_map)
scene_stat = scene_stat.sort_values("ratio", ascending=False)  # 按比例降序排列
scene_stat.to_csv("scene_ratio.csv", index=False, encoding="utf-8-sig")
print("💾 已保存 scene_ratio.csv")

# 打印场景统计数值
print(f"\n📊 场景分布统计（决策场景下玄学内容占比）:")
print("   " + "="*70)
print(f"   {'场景（中文）':15s} {'场景（英文）':20s} {'数量':8s} {'占比':12s} {'排名':6s}")
print("   " + "-"*70)
total_count = scene_stat["count"].sum()
for rank, row in enumerate(scene_stat.itertuples(), 1):
    scene_cn = row.scene
    scene_en = row.scene_en
    count = row.count
    ratio = row.ratio * 100
    print(f"   {scene_cn:15s} {scene_en:20s} {count:8d} {ratio:11.2f}% #{rank:<5d}")
print("   " + "="*70)
print(f"   总样本数: {total_count}")
max_ratio_scene = scene_stat.iloc[0]
print(f"   占比最高场景: {max_ratio_scene['scene']} ({max_ratio_scene['scene_en']}) - {max_ratio_scene['ratio']*100:.2f}%")
print("   说明: 占比越高，说明该场景下用户主动提及玄学内容的比例越高")

# -------------------------
# 5. Scene Visualization
# -------------------------
# Bar Chart
plt.figure(figsize=(8,5))
plt.bar(scene_stat["scene_en"], scene_stat["ratio"], color="skyblue")
plt.xlabel("Decision Scenario")
plt.ylabel("Proportion of Esoteric Content")
plt.title("Proportion of Esoteric Content by Decision Scenario")
plt.tight_layout()
plt.savefig("scene_ratio_bar_en.png", dpi=300)
print("\n💾 已保存场景比例柱状图: scene_ratio_bar_en.png")
print("   📊 图片中的具体数值（按占比降序）:")
for row in scene_stat.itertuples():
    print(f"      {row.scene_en:20s}: {row.ratio*100:5.2f}% ({row.count}个视频)")
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
plt.pie(
    scene_stat["ratio"],
    labels=scene_stat["scene_en"],
    autopct="%.1f%%",
    startangle=140
)
plt.title("Distribution of Esoteric Content Across Decision Scenarios")
plt.tight_layout()
plt.savefig("scene_ratio_pie_en.png", dpi=300)
print("\n💾 已保存场景分布饼图: scene_ratio_pie_en.png")
print("   📊 图片中的具体数值:")
for row in scene_stat.itertuples():
    print(f"      {row.scene_en:20s}: {row.ratio*100:5.2f}% ({row.count}个视频)")
plt.show()

# -------------------------
# 6. Other 子场景处理
# -------------------------
keywords_map = {
    "Consumption": ["钱", "投资", "理财", "买", "消费", "购物", "花费"],
    "Health": ["身体", "健康", "睡眠", "情绪", "心情", "心理", "养生", "运动", "焦虑", "心灵", "压力"],
    "Identity": ["mbti", "性格", "自我", "人格", "心理测试", "自我成长", "心灵", "内在"]
}

def classify_other_subscene(title):
    title = str(title).lower()
    words = list(jieba.cut(title))
    for subscene, kws in keywords_map.items():
        for kw in kws:
            if kw in words or kw in title:
                return subscene
    return "Misc"

df.loc[df["scene"] == "其他", "other_subscene"] = df.loc[df["scene"] == "其他", "title"].apply(classify_other_subscene)

# 统计 Other 子场景
other_stat = (
    df[df["scene"] == "其他"]
      .groupby("other_subscene")
      .size()
      .reset_index(name="count")
)
other_stat["ratio"] = other_stat["count"] / other_stat["count"].sum()
other_stat = other_stat.sort_values("ratio", ascending=False)  # 按比例降序排列
other_stat.to_csv("other_subscene_ratio.csv", index=False, encoding="utf-8-sig")
df.to_csv("bilibili_videos_with_scene_subscene.csv", index=False, encoding="utf-8-sig")
print("💾 已保存 Other 子场景数据和统计结果")

# 打印Other子场景统计数值
other_total = other_stat["count"].sum()
other_total_all = len(df[df["scene"] == "其他"])
print(f"\n📊 Other类别子场景分布统计:")
print("   " + "="*60)
print(f"   {'子场景':20s} {'数量':8s} {'占比（Other内）':15s} {'排名':6s}")
print("   " + "-"*60)
if len(other_stat) > 0:
    for rank, row in enumerate(other_stat.itertuples(), 1):
        subscene = row.other_subscene
        count = row.count
        ratio = row.ratio * 100
        print(f"   {subscene:20s} {count:8d} {ratio:14.2f}% #{rank:<5d}")
    print("   " + "="*60)
    print(f"   Other类别总视频数: {other_total_all}")
    print(f"   已分类视频数: {other_total}")
    print(f"   未分类（Misc）: {other_total_all - other_total if other_total_all > other_total else 0}")
    if len(other_stat) > 0:
        max_ratio_subscene = other_stat.iloc[0]
        print(f"   占比最高子场景: {max_ratio_subscene['other_subscene']} - {max_ratio_subscene['ratio']*100:.2f}%")
else:
    print("   无Other类别子场景数据")
    print("   " + "="*60)

# -------------------------
# 7. Other 子场景可视化
# -------------------------
# Bar Chart
if len(other_stat) > 0:
    plt.figure(figsize=(8,5))
    plt.bar(other_stat["other_subscene"], other_stat["ratio"], color="lightcoral")
    plt.xlabel("Other Subscene")
    plt.ylabel("Proportion")
    plt.title("Proportion of Esoteric Content in Other Subscenes")
    plt.tight_layout()
    plt.savefig("other_subscene_bar_en.png", dpi=300)
    print("\n💾 已保存Other子场景柱状图: other_subscene_bar_en.png")
    print("   📊 图片中的具体数值（按占比降序）:")
    for row in other_stat.itertuples():
        print(f"      {row.other_subscene:20s}: {row.ratio*100:5.2f}% ({row.count}个视频)")
    plt.show()
else:
    print("\n⚠️ 无Other子场景数据，跳过Other子场景柱状图生成")

# Pie Chart
if len(other_stat) > 0:
    plt.figure(figsize=(6,6))
    plt.pie(
        other_stat["ratio"],
        labels=other_stat["other_subscene"],
        autopct="%.1f%%",
        startangle=140,
        colors=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
    )
    plt.title("Distribution of Esoteric Content Across Other Subscenes")
    plt.tight_layout()
    plt.savefig("other_subscene_pie_en.png", dpi=300)
    print("\n💾 已保存Other子场景饼图: other_subscene_pie_en.png")
    print("   📊 图片中的具体数值:")
    for row in other_stat.itertuples():
        print(f"      {row.other_subscene:20s}: {row.ratio*100:5.2f}% ({row.count}个视频)")
    plt.show()
else:
    print("\n⚠️ 无Other子场景数据，跳过Other子场景饼图生成")

print("✅ Q1 Analysis Completed: All scenes + Other subscenes processed and visualized.")
