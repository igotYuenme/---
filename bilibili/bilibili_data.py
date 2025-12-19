import requests
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# ⚠️ 用你浏览器里的 cookie（不需要全部）
COOKIES = {
    "SESSDATA": "fe6b47f8%2C1781276922%2C9218b%2Ac1CjCj2ylimx2cRdVzjuVh3-dT6p_21q9h88Jk2qpoSwgIMS_h10xgu3tKqlkMuDwpVqISVlJqN2FhTWpPUWJZdE5ELVh0Q1o1a2ZSSndTTV8yQXdYNlY3UEZaNmtvck1VVXpVeHlVd05iSUowN2xzUlBnV1J6ZHZDVDNOVlNkRjZobzRYRUZySV93IIEC",
    "bili_jct": "4cc35775f5ade0d0803a91688acc8869",
}

keywords = [
    '星象分析',      # 替代 星座
    '抽牌建议',      # 替代 塔罗
    '水逆',
    '运势',
    '星盘运势',      # 替代 星盘
    'MBTI',
    '显化',
    '吸引力法则',
    '考前 建议',     # 替代 考前 占卜
    '分手 建议',     # 替代 分手 塔罗
    '面试 建议',     # 替代 面试 星座
    '考试 运势'
]


def search_bilibili(keyword, pages=3):
    print(f"[DEBUG] 进入 search_bilibili(): {keyword}")
    results = []
    for page in range(1, pages + 1):
        url = "https://api.bilibili.com/x/web-interface/search/type"
        params = {
            "search_type": "video",
            "keyword": keyword,
            "page": page
        }

        try:
            r = requests.get(
                url,
                headers=HEADERS,
                cookies=COOKIES,
                params=params,
                timeout=10
            )

            # 🚑 兜底判断
            if r.status_code != 200 or not r.text.strip():
                print(f"[WARN] {keyword} page={page} 返回空内容")
                continue

            data = r.json()

            items = data.get("data", {}).get("result", [])
            if not items:
                print(f"[INFO] {keyword} page={page} 无结果")
                continue

            for v in items:
                results.append({
                    "keyword": keyword,
                    "title": v.get("title", "").replace("<em class=\"keyword\">", "").replace("</em>", ""),
                    "up": v.get("author", ""),
                    "play": v.get("play", 0),
                    "danmu": v.get("danmaku", 0),
                    "pubdate": v.get("pubdate", 0),
                    "bvid": v.get("bvid", ""),
                    "link": f"https://www.bilibili.com/video/{v.get('bvid','')}"
                })

            time.sleep(random.uniform(1.5, 3))

        except Exception as e:
            print(f"[ERROR] {keyword} page={page} 异常：{e}")
            continue

    return results

if __name__ == "__main__":
    print("✅ 进入主程序")

    all_data = []

    for kw in keywords:
        print(f"🔍 搜索：{kw}")
        data = search_bilibili(kw, pages=2)
        print(f"   👉 返回 {len(data)} 条")
        all_data.extend(data)

    print(f"📊 总计抓取视频数：{len(all_data)}")

    import pandas as pd
    df = pd.DataFrame(all_data)

    print(df.head())

    df.to_csv("bilibili_videos.csv", index=False, encoding="utf-8-sig")
    print("💾 已保存 bilibili_videos.csv")
