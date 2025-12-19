import requests
import pandas as pd
import time

# -----------------------------
# 配置
# -----------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.7499.41 Safari/537.36",
    "cookie": "gid=yjDJySdS0Y0dyjDJySd0fhDVdD2TWx12YVADJU0h6YSUdD286iE6fd888yj22q48j0fKi24d;loadts=1765798751723;sec_poison_id=9aeb7f24-dc32-4135-97bd-f3e9a8369cbe;unread=9aeb7f24-dc32-4135-97bd-f3e9a8369cbe;web_session=04006979c29a22dc9a3312bb0a3b4bf4105ef0;webBuild=5.0.6;webld=f36c57760a55a5aae93dbafa56a308c8;websectiga=634d3ad75ffb42a2ade2c5e1705a73c845837578aeb31ba0e442d75c648da36a; xsecappid=xhs-pc-web",
    "Referer": "https://www.xiaohongshu.com/"
}

SEARCH_URL = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"

KEYWORDS = [
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

MAX_PAGES = 3  # 每个关键词抓取页数，每页大约 10 条笔记
SLEEP_TIME = 1  # 请求间隔，避免封 IP

# -----------------------------
# 小红书笔记抓取函数
# -----------------------------
def fetch_notes(keyword, pages=MAX_PAGES):
    notes = []
    for page in range(1, pages + 1):
        # 关键：构建一个包含更多参数的请求
        params = {
            "keyword": keyword,
            "page": page,
            "page_size": 20,  # 每页数量可尝试调整
            "sort": "general",  # 排序方式：general(综合) time_desc(最新)
            # 根据逆向分析，可能还需要以下参数：
            # "search_id": "生成一个随机ID",
            # "source": "input",
            # "need_web_search": "true",
        }
        try:
            resp = requests.get(SEARCH_URL, headers=HEADERS, params=params, timeout=15)
            print(f"调试信息: 状态码 {resp.status_code}, 响应长度 {len(resp.text)}")
            print(f"响应前100字符: {resp.text[:100]}") # 查看响应内容，帮助判断
            
            # 尝试解析JSON
            data = resp.json()
            # 注意：数据层级结构很可能已经改变，需要你根据实际返回的JSON结构调整
            items = data.get("data", {}).get("items", [])  # 这里只是假设，需调整
            if not items:
                # 也可能在别的字段里
                items = data.get("data", {}).get("notes", [])
                items = data.get("data", {}).get("list", [])
            
            for item in items:
                note = {
                    "keyword": keyword,
                    "title": item.get("title") or item.get("note_card", {}).get("display_title", ""),
                    "desc": item.get("desc") or item.get("note_card", {}).get("desc", ""), # 增加内容字段
                    "author": item.get("user", {}).get("nickname", ""),
                    "likes": item.get("likes") or item.get("note_card", {}).get("interact_info", {}).get("liked_count", 0),
                    "comments": item.get("comments") or item.get("note_card", {}).get("interact_info", {}).get("comment_count", 0),
                    "favorites": item.get("favorites") or item.get("note_card", {}).get("interact_info", {}).get("collected_count", 0),
                    "note_id": item.get("id", ""),
                    "link": f"https://www.xiaohongshu.com/explore/{item.get('id', '')}"
                }
                notes.append(note)
            print(f"[INFO] {keyword} page={page} 获取 {len(items)} 条笔记")
            time.sleep(SLEEP_TIME)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 网络请求失败: {e}")
        except ValueError as e:
            print(f"[ERROR] 解析JSON失败，响应可能不是JSON格式。原始响应开头: {resp.text[:200]}")
            # 这里很可能是遇到了反爬，返回了验证页面
        except Exception as e:
            print(f"[ERROR] 其他错误: {e}")
    return notes


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    all_notes = []
    for kw in KEYWORDS:
        print(f"🔍 搜索关键词: {kw}")
        notes = fetch_notes(kw, pages=MAX_PAGES)
        all_notes.extend(notes)

    if all_notes:
        df = pd.DataFrame(all_notes)
        df.to_csv("xiaohongshu_notes.csv", index=False, encoding="utf-8-sig")
        print(f"💾 已保存 {len(df)} 条笔记至 xiaohongshu_notes.csv")
    else:
        print("⚠️ 未抓取到任何笔记，请检查 Cookie 或关键词")
