# ======================================
# 收集UP主本人的视频内容
# 功能：专门收集指定UP主（如"龙女塔罗"）发布的视频
# ======================================

import requests
import time
import random
import pandas as pd
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# ⚠️ 用你浏览器里的 cookie
COOKIES = {
    "SESSDATA": "fe6b47f8%2C1781276922%2C9218b%2Ac1CjCj2ylimx2cRdVzjuVh3-dT6p_21q9h88Jk2qpoSwgIMS_h10xgu3tKqlkMuDwpVqISVlJqN2FhTWpPUWJZdE5ELVh0Q1o1a2ZSSndTTV8yQXdYNlY3UEZaNmtvck1VVXpVeHlVd05iSUowN2xzUlBnV1J6ZHZDVDNOVlNkRjZobzRYRUZySV93IIEC",
    "bili_jct": "4cc35775f5ade0d0803a91688acc8869",
}

def search_up_videos(up_name, max_pages=20):
    """
    搜索UP主的视频（通过搜索UP主名称，然后过滤作者字段）
    
    Args:
        up_name: UP主名称，如"龙女塔罗"
        max_pages: 最大搜索页数
    
    Returns:
        list: 视频数据列表
    """
    print(f"🔍 开始收集UP主 '{up_name}' 的视频...")
    all_results = []
    seen_bvids = set()  # 用于去重
    
    # 搜索UP主名称
    for page in range(1, max_pages + 1):
        url = "https://api.bilibili.com/x/web-interface/search/type"
        params = {
            "search_type": "video",
            "keyword": up_name,  # 搜索UP主名称
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
            
            if r.status_code != 200 or not r.text.strip():
                print(f"  [WARN] 第{page}页返回空内容，停止搜索")
                break
            
            data = r.json()
            
            # 检查API返回状态
            if data.get('code') != 0:
                print(f"  [WARN] API返回错误: {data.get('message', '未知错误')}")
                break
            
            items = data.get("data", {}).get("result", [])
            if not items:
                print(f"  [INFO] 第{page}页无结果，停止搜索")
                break
            
            page_count = 0
            for v in items:
                author = v.get("author", "")
                bvid = v.get("bvid", "")
                
                # 只收集UP主本人的视频（精确匹配作者名）
                if author == up_name and bvid and bvid not in seen_bvids:
                    seen_bvids.add(bvid)
                    all_results.append({
                        "keyword": f"UP主:{up_name}",  # 标记为UP主视频
                        "title": v.get("title", "").replace("<em class=\"keyword\">", "").replace("</em>", ""),
                        "up": author,
                        "play": v.get("play", 0),
                        "danmu": v.get("danmaku", 0),
                        "pubdate": v.get("pubdate", 0),
                        "bvid": bvid,
                        "link": f"https://www.bilibili.com/video/{bvid}"
                    })
                    page_count += 1
            
            print(f"  第{page}页: 找到{page_count}个UP主视频 (累计: {len(all_results)})")
            
            # 如果这一页没有找到UP主视频，可能已经到底了
            if page_count == 0:
                print(f"  连续多页无UP主视频，停止搜索")
                break
            
            time.sleep(random.uniform(2, 4))  # 避免请求过快
            
        except Exception as e:
            print(f"  [ERROR] 第{page}页异常: {e}")
            continue
    
    print(f"✅ 共收集到 {len(all_results)} 个UP主 '{up_name}' 的视频")
    return all_results

def collect_up_videos(up_name="龙女塔罗", max_pages=30, save_file=None):
    """
    收集UP主的视频并保存到CSV文件
    
    Args:
        up_name: UP主名称
        max_pages: 最大搜索页数
        save_file: 保存文件名，如果为None则自动生成
    """
    print("=" * 70)
    print(f"收集UP主视频内容: {up_name}")
    print("=" * 70)
    print()
    
    # 收集视频
    videos = search_up_videos(up_name, max_pages=max_pages)
    
    if len(videos) == 0:
        print("❌ 未收集到任何视频，请检查:")
        print("   1. UP主名称是否正确")
        print("   2. 网络连接是否正常")
        print("   3. Cookie是否有效")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(videos)
    
    # 打印统计信息
    print(f"\n📊 收集结果统计:")
    print(f"   视频总数: {len(df)}")
    if 'play' in df.columns:
        total_views = pd.to_numeric(df['play'], errors='coerce').sum()
        avg_views = pd.to_numeric(df['play'], errors='coerce').mean()
        print(f"   总播放量: {total_views:,.0f}")
        print(f"   平均播放量: {avg_views:,.0f}")
    
    if 'danmu' in df.columns:
        total_danmu = pd.to_numeric(df['danmu'], errors='coerce').sum()
        print(f"   总弹幕数: {total_danmu:,.0f}")
    
    # 保存文件
    if save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = f"{up_name}_videos_{timestamp}.csv"
    
    df.to_csv(save_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 已保存到: {save_file}")
    
    return df

if __name__ == "__main__":
    # 配置参数
    UP_NAME = "龙女塔罗"
    MAX_PAGES = 30  # 增加页数以获取更多视频
    
    df = collect_up_videos(UP_NAME, max_pages=MAX_PAGES)
    
    if df is not None:
        print(f"\n✅ 收集完成！")
        print(f"   文件已保存，可在longnv.py中使用")

