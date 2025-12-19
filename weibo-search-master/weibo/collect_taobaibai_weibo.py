# ======================================
# 收集陶白白本人的微博内容
# 功能：专门收集指定博主（如"陶白白"）发布的微博
# ======================================

import requests
import json
import time
import random
from datetime import datetime
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings()

COOKIE = 'WEIBOCN_FROM=1110006030; _T_WM=99895283787; SCF=At_bl9yByv0ENbFBSKWHytS7iH19oSoSfd_9dSXjyskqMABoeCjyLnQJ1gvzU8bXVoijHRwx32Q3KCGyQGa4Du8.; SUB=_2A25EOFU3DeRhGeBM6lUQ-C_Nzz-IHXVnNOj_rDV6PUJbktANLU7ckW1NRDAG61lJHxT9WgTcouUX7_VvbeuFW2Id; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWB.cA_nFF2RAP6OksIX5YY5NHD95Qceo2NeKnpeKB0Ws4Dqcj6i--ciKy2iKysi--fiKysi-8Wi--fi-z7iKysi--4i-zpi-ihi--fiKLhiKnci--fiKLhiKnci--fiKLhiKnc; SSOLoginState=1765549415; ALF=1768141415; MLOGIN=1; XSRF-TOKEN=f5f68c; M_WEIBOCN_PARAMS=lfid%3D102803%26luicode%3D20000174%26uicode%3D20000174'

headers = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 10; Pixel 5) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36',
    'Cookie': COOKIE,
    'Referer': 'https://m.weibo.cn/'
}

def find_user_uid(blogger_name, session, max_search_pages=5):
    """
    通过搜索找到用户的UID
    
    Args:
        blogger_name: 博主名称
        session: requests会话对象
        max_search_pages: 最大搜索页数
    
    Returns:
        str: 用户UID，如果找不到返回None
    """
    print(f"  🔍 步骤1: 搜索用户 '{blogger_name}' 以获取UID...")
    
    # 用于调试：收集所有找到的用户名
    all_found_authors = []
    
    # 尝试搜索多页，找到一条该用户发布的微博
    for page in range(1, max_search_pages + 1):
        params = {
            'containerid': f'100103type=1&q={blogger_name}',
            'page_type': 'searchall',
            'page': page
        }
        
        try:
            resp = session.get(
                'https://m.weibo.cn/api/container/getIndex',
                headers=headers,
                params=params,
                timeout=20,
                verify=False
            )
            
            if resp.status_code != 200:
                continue
            
            data = resp.json()
            if data.get('ok') != 1:
                continue
            
            cards = data.get('data', {}).get('cards', [])
            page_authors = []
            
            for card in cards:
                if card.get('card_type') == 9:
                    mblog = card.get('mblog')
                    if not mblog:
                        continue
                    
                    user = mblog.get('user', {})
                    author_name = user.get('screen_name', '')
                    user_id = user.get('id')
                    
                    # 收集所有作者名用于调试（只在前3页）
                    if page <= 3 and author_name:
                        page_authors.append(author_name)
                    
                    # 检查是否是目标用户（支持多种匹配方式）
                    # 1. 精确匹配
                    # 2. 包含匹配（博主名称包含在作者名中）
                    # 3. 作者名包含在博主名称中
                    # 4. 包含"陶"和"白"（中文匹配）
                    # 5. 包含"tao"和"bai"（拼音匹配）
                    is_target = (
                        author_name == blogger_name or
                        blogger_name in author_name or
                        author_name in blogger_name or
                        (author_name and '陶' in author_name and '白' in author_name) or
                        (author_name and 'tao' in author_name.lower() and 'bai' in author_name.lower())
                    )
                    
                    if is_target and user_id:
                        print(f"  ✅ 在第{page}页找到用户 '{author_name}'，UID: {user_id}")
                        return str(user_id)
            
            # 显示每页找到的作者（前3页）
            if page <= 3 and page_authors:
                unique_authors = list(set(page_authors))[:15]  # 显示前15个不同的作者
                all_found_authors.extend(unique_authors)
                print(f"  [调试] 第{page}页找到的作者（前15个）: {unique_authors}")
            
            time.sleep(random.uniform(1, 2))  # 避免请求过快
            
        except Exception as e:
            print(f"  [WARN] 搜索第{page}页时出错: {e}")
            continue
    
    # 显示所有找到的相关用户名
    if all_found_authors:
        unique_all = list(set(all_found_authors))
        # 筛选可能相关的用户名（包含"陶"或"白"）
        related_authors = [a for a in unique_all if '陶' in a or '白' in a or 'tao' in a.lower() or 'bai' in a.lower()]
        if related_authors:
            print(f"\n  💡 找到的可能相关用户（前20个）:")
            for author in related_authors[:20]:
                print(f"     - {author}")
    
    print(f"\n  ⚠️ 在前{max_search_pages}页搜索结果中未找到用户 '{blogger_name}' 的微博")
    print(f"  💡 提示：可以尝试使用博主的实际用户名，或者直接在代码中设置UID")
    return None


def get_user_timeline(user_id, blogger_name, session, max_pages=30):
    """
    使用用户时间线API获取用户的微博
    
    Args:
        user_id: 用户UID
        blogger_name: 博主名称（用于标记）
        session: requests会话对象
        max_pages: 最大页数
    
    Returns:
        list: 微博数据列表
    """
    print(f"  📝 步骤2: 使用时间线API获取用户微博（UID: {user_id}）...")
    all_weibos = []
    seen_ids = set()
    
    # 用户时间线的containerid格式：107603{uid}
    containerid = f'107603{user_id}'
    last_weibo_id = None
    
    for page in range(1, max_pages + 1):
        params = {
            'containerid': containerid,
            'page': page
        }
        
        # 如果上一页有微博ID，使用since_id翻页
        if last_weibo_id:
            params['since_id'] = last_weibo_id
        
        try:
            resp = session.get(
                'https://m.weibo.cn/api/container/getIndex',
                headers=headers,
                params=params,
                timeout=20,
                verify=False
            )
            
            if resp.status_code != 200:
                print(f"  [WARN] 第{page}页HTTP {resp.status_code}，停止")
                break
            
            data = resp.json()
            if data.get('ok') != 1:
                print(f"  [WARN] API返回错误: {data.get('msg', '未知错误')}")
                break
            
            cards = data.get('data', {}).get('cards', [])
            if not cards:
                print(f"  [INFO] 第{page}页无结果，停止")
                break
            
            page_count = 0
            for card in cards:
                if card.get('card_type') == 9:
                    mblog = card.get('mblog')
                    if not mblog:
                        continue
                    
                    weibo_id = mblog.get('id')
                    if weibo_id and weibo_id not in seen_ids:
                        seen_ids.add(weibo_id)
                        user = mblog.get('user', {})
                        author_name = user.get('screen_name', blogger_name)
                        
                        all_weibos.append({
                            'platform': 'weibo',
                            'keyword': f'博主:{blogger_name}',
                            'id': weibo_id,
                            'text': mblog.get('text', ''),
                            'created_at': mblog.get('created_at'),
                            'reposts': mblog.get('reposts_count', 0),
                            'comments': mblog.get('comments_count', 0),
                            'likes': mblog.get('attitudes_count', 0),
                            'user': author_name
                        })
                        page_count += 1
                        last_weibo_id = weibo_id
            
            print(f"    第{page}页: 获取{page_count}条微博 (累计: {len(all_weibos)})")
            
            if page_count == 0:
                break
            
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"  [ERROR] 第{page}页异常: {e}")
            continue
    
    return all_weibos


def search_blogger_weibo(blogger_name, max_pages=30, user_id=None):
    """
    收集博主的微博（先找到用户UID，然后使用时间线API）
    
    Args:
        blogger_name: 博主名称，如"陶白白"或"陶白白Sensei"
        max_pages: 最大页数
        user_id: 用户UID（可选，如果提供则直接使用，不搜索）
    
    Returns:
        list: 微博数据列表
    """
    print(f"🔍 开始收集博主 '{blogger_name}' 的微博...")
    all_weibos = []
    
    # 配置会话
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 步骤1: 找到用户UID（如果未提供）
    if user_id:
        print(f"  ✅ 使用提供的UID: {user_id}")
        final_user_id = str(user_id)
    else:
        final_user_id = find_user_uid(blogger_name, session)
        if not final_user_id:
            print(f"  ❌ 无法找到用户 '{blogger_name}' 的UID")
            print(f"  💡 如果知道用户的UID，可以在代码中直接设置 user_id 参数")
            return []
    
    # 步骤2: 使用时间线API获取微博
    all_weibos = get_user_timeline(final_user_id, blogger_name, session, max_pages)
    
    print(f"✅ 共收集到 {len(all_weibos)} 条博主 '{blogger_name}' 的微博")
    return all_weibos


def collect_blogger_weibo(blogger_name="陶白白", max_pages=30, save_file=None, user_id=None):
    """
    收集博主的微博并保存到JSON文件
    
    Args:
        blogger_name: 博主名称
        max_pages: 最大搜索页数
        save_file: 保存文件名，如果为None则自动生成
        user_id: 用户UID（可选，如果提供则直接使用，不搜索）
    """
    print("=" * 70)
    print(f"收集博主微博内容: {blogger_name}")
    print("=" * 70)
    print()
    
    # 收集微博
    weibos = search_blogger_weibo(blogger_name, max_pages=max_pages, user_id=user_id)
    
    if len(weibos) == 0:
        print("❌ 未收集到任何微博，请检查:")
        print("   1. 博主名称是否正确")
        print("   2. 网络连接是否正常")
        print("   3. Cookie是否有效")
        return None
    
    # 打印统计信息
    print(f"\n📊 收集结果统计:")
    print(f"   微博总数: {len(weibos)}")
    
    total_reposts = sum(w.get('reposts', 0) for w in weibos)
    total_comments = sum(w.get('comments', 0) for w in weibos)
    total_likes = sum(w.get('likes', 0) for w in weibos)
    
    if total_reposts > 0:
        print(f"   总转发数: {total_reposts:,.0f}")
    if total_comments > 0:
        print(f"   总评论数: {total_comments:,.0f}")
    if total_likes > 0:
        print(f"   总点赞数: {total_likes:,.0f}")
    
    # 保存文件
    if save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = f"{blogger_name}_weibo_{timestamp}.json"
    
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(weibos, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 已保存到: {save_file}")
    
    return weibos

if __name__ == "__main__":
    # 配置参数
    BLOGGER_NAME = "陶白白Sensei"
    MAX_PAGES = 30  # 增加页数以获取更多微博
    USER_ID = "6003325152"  # 用户的UID
    
    # 如果设置了USER_ID，则直接使用；否则会尝试搜索
    weibos = collect_blogger_weibo(BLOGGER_NAME, max_pages=MAX_PAGES, user_id=USER_ID)
    
    if weibos is not None:
        print(f"\n✅ 收集完成！")
        print(f"   文件已保存，可在taobaibai.py中使用")

