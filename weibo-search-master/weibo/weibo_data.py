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

KEYWORDS = [
    # 核心关键词（高频）
    '星象分析',
    '抽牌建议',
    '水逆',
    '运势',
    '星盘运势',
    'MBTI',
    '显化',
    '吸引力法则',
    # 学业/职业相关（心理慰藉型）
    '考前 建议',
    '考试 运势',
    '面试 建议',
    '求职 运势',
    '考研 运势',
    '毕业 建议',
    '实习 建议',
    'offer 运势',
    '论文 建议',
    # 情感相关（娱乐型）
    '分手 建议',
    '复合 运势',
    '恋爱 建议',
    '桃花 运势',
    '前任 建议',
    '暧昧 建议',
    # 扩展关键词
    '塔罗牌',
    '占卜',
    '星座运势',
    '心理测试',
    '星盘',
    '塔罗',
    '占星',
    '运势分析',
    '水逆期',
    '吸引力',
    '显化法则',
    'MBTI测试',
    '性格测试',
    '情感咨询',
    '学业咨询',
    '职业规划',
    '面试技巧',
    '考试焦虑',
    '分手复合',
    '恋爱技巧',
    # 新增关键词（扩大覆盖范围）
    '陶白白',
    '星座',
    '情感分析',
    '心理分析',
    '性格分析',
    '人格测试',
    '心理慰藉',
    '情感指导',
    '学业指导',
    '职业建议',
    '面试指导',
    '考试建议',
    '复合建议',
    '分手建议',
    '恋爱指导',
    '情感解惑',
    '心理支持',
    '情绪管理',
    '焦虑缓解',
    '压力缓解',
    '情感答疑',
    '心理答疑',
    '运势预测',
    '未来预测',
    '人生建议',
    '生活建议',
    '决策建议',
    '选择困难',
    '迷茫',
    '困惑',
    '求助',
    '咨询',
    '分析',
    '解读',
    '指引',
    '指导',
    '建议',
    '方法',
    '技巧',
    '策略'
]


MAX_PAGES = 30         # 每个关键词最多抓多少页（增加到30页以获得更多数据）
TARGET_TOTAL = 3000    # 目标总数（提高到3000条以获得更充足的数据）
MIN_TOTAL = 1500       # 最少抓取数量（提高到1500条）
EMPTY_LIMIT = 3        # 连续空页数限制（增加到3）

# Session + Retry
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

all_weibos = []

for kw_idx, keyword in enumerate(KEYWORDS, 1):
    print(f'\n===== Keyword ({kw_idx}/{len(KEYWORDS)}): {keyword} =====')
    empty_pages = 0
    page_count = 0
    since_id = None
    search_ssid = None
    search_vsid = None
    containerid_base = None
    last_weibo_id = None  # 保存最后一个微博ID，用于翻页

    while page_count < MAX_PAGES and len(all_weibos) < TARGET_TOTAL:
        # 构建请求参数
        if page_count == 0:
            # 第一页使用 containerid
            params = {
                'containerid': f'100103type=1&q={keyword}',
                'page_type': 'searchall'
            }
        else:
            # 后续页面：优先使用最后一个微博ID作为since_id（最可靠）
            if last_weibo_id:
                params = {
                    'containerid': f'100103type=1&q={keyword}',
                    'page_type': 'searchall',
                    'since_id': last_weibo_id
                }
                if search_ssid:
                    params['search_ssid'] = search_ssid
                if search_vsid:
                    params['search_vsid'] = search_vsid
                print(f'  [调试] 使用since_id={last_weibo_id}翻页')
            elif containerid_base:
                # 使用完整的containerid + page
                params = {
                    'containerid': containerid_base,
                    'page_type': 'searchall',
                    'page': page_count + 1
                }
                if search_ssid:
                    params['search_ssid'] = search_ssid
                if search_vsid:
                    params['search_vsid'] = search_vsid
                print(f'  [调试] 使用containerid_base + page={page_count + 1}')
            elif since_id:
                # 使用API返回的since_id
                params = {
                    'containerid': f'100103type=1&q={keyword}',
                    'page_type': 'searchall',
                    'since_id': since_id
                }
                if search_ssid:
                    params['search_ssid'] = search_ssid
                if search_vsid:
                    params['search_vsid'] = search_vsid
                print(f'  [调试] 使用API返回的since_id={since_id}')
            else:
                # 最后尝试：使用page参数
                params = {
                    'containerid': f'100103type=1&q={keyword}',
                    'page_type': 'searchall',
                    'page': page_count + 1
                }
                if search_ssid:
                    params['search_ssid'] = search_ssid
                if search_vsid:
                    params['search_vsid'] = search_vsid
                print(f'  [调试] 使用page={page_count + 1}（最后尝试）')

        success = False
        for attempt in range(3):
            try:
                resp = session.get(
                    'https://m.weibo.cn/api/container/getIndex',
                    headers=headers,
                    params=params,
                    timeout=20,
                    verify=False
                )
                
                # 检查响应状态
                if resp.status_code != 200:
                    print(f'  HTTP {resp.status_code}, retrying...')
                    time.sleep(random.uniform(3, 6))
                    continue
                
                data = resp.json()
                
                # 检查API返回状态
                if data.get('ok') != 1:
                    print(f'  API返回错误: {data.get("msg", "未知错误")}')
                    if data.get('msg') and '频繁' in data.get('msg', ''):
                        print('  ⚠️ 可能触发频率限制，等待更长时间...')
                        time.sleep(random.uniform(30, 60))
                    break
                
                # 保存会话信息（第一页）
                if page_count == 0:
                    cardlist_info = data.get('data', {}).get('cardlistInfo', {})
                    # 保存搜索会话ID
                    search_ssid = cardlist_info.get('search_ssid')
                    search_vsid = cardlist_info.get('search_vsid')
                    containerid_base = cardlist_info.get('containerid')
                    
                    # 调试信息
                    print(f'  [调试] total: {cardlist_info.get("total", 0)} 条结果')
                    print(f'  [调试] page_size: {cardlist_info.get("page_size", 0)}')
                    if search_ssid:
                        print(f'  [调试] search_ssid: {search_ssid[:20]}...')
                    if search_vsid:
                        print(f'  [调试] search_vsid: {search_vsid[:20]}...')
                    if containerid_base:
                        print(f'  [调试] containerid: {containerid_base[:50]}...')
                
                success = True
                break
            except json.JSONDecodeError as e:
                print(f'  JSON解析失败: {e}')
                print(f'  响应内容前200字符: {resp.text[:200]}')
                time.sleep(random.uniform(5, 10))
            except Exception as e:
                print(f'  Attempt {attempt+1} failed: {e}')
                time.sleep(random.uniform(3, 6))

        if not success:
            print('  Skipped this page due to repeated errors.')
            empty_pages += 1
            if empty_pages >= EMPTY_LIMIT:
                break
            continue

        cards = data.get('data', {}).get('cards', [])
        count = 0
        current_page_last_id = None  # 当前页最后一个微博ID
        
        for card in cards:
            if card.get('card_type') == 9:
                mblog = card.get('mblog')
                if not mblog:
                    continue
                
                # 去重：检查是否已存在相同ID的微博
                weibo_id = mblog.get('id')
                if any(w.get('id') == weibo_id for w in all_weibos):
                    continue
                
                all_weibos.append({
                    'platform': 'weibo',
                    'keyword': keyword,
                    'id': weibo_id,
                    'text': mblog.get('text', ''),
                    'created_at': mblog.get('created_at'),
                    'reposts': mblog.get('reposts_count', 0),
                    'comments': mblog.get('comments_count', 0),
                    'likes': mblog.get('attitudes_count', 0),
                    'user': mblog.get('user', {}).get('screen_name', '')
                })
                count += 1
                current_page_last_id = weibo_id  # 更新当前页最后一个微博ID
        
        # 更新全局last_weibo_id（用于下次翻页）
        if current_page_last_id:
            last_weibo_id = current_page_last_id

        print(f'  Page {page_count+1}: {count} posts (累计: {len(all_weibos)}/{TARGET_TOTAL})')

        if count == 0:
            empty_pages += 1
            if empty_pages >= EMPTY_LIMIT:
                print(f'  连续{EMPTY_LIMIT}页为空，停止抓取该关键词')
                break
        else:
            empty_pages = 0

        # 获取下一页信息
        cardlist_info = data.get('data', {}).get('cardlistInfo', {})
        current_page = cardlist_info.get('page', page_count + 1)
        
        # 确保类型转换（API可能返回字符串）
        try:
            total_results = int(cardlist_info.get('total', 0))
        except (ValueError, TypeError):
            total_results = 0
        
        try:
            page_size = int(cardlist_info.get('page_size', 10))
        except (ValueError, TypeError):
            page_size = 10
        
        try:
            current_page = int(current_page)
        except (ValueError, TypeError):
            current_page = page_count + 1
        
        # 计算总页数
        if total_results > 0 and page_size > 0:
            total_pages = (total_results + page_size - 1) // page_size
            print(f'  [信息] 当前页: {current_page}/{total_pages}, 每页: {page_size}条')
        
        # 尝试获取 since_id
        since_id = cardlist_info.get('since_id')
        if not since_id:
            since_id = cardlist_info.get('since_id_str') or cardlist_info.get('next_cursor')
        
        # 检查是否还有更多页面
        if total_results > 0:
            estimated_current = (current_page - 1) * page_size + count
            if estimated_current >= total_results:
                print(f'  已抓取 {estimated_current}/{total_results}，没有更多数据了')
                break
        elif count == 0 and page_count > 0:
            # 如果连续多页都是0条，可能没有更多数据了
            print(f'  连续多页无数据，可能已抓取完毕')
            break

        page_count += 1
        
        # 动态调整等待时间：接近目标时加快速度
        if len(all_weibos) < TARGET_TOTAL * 0.8:
            time.sleep(random.uniform(5, 10))
        else:
            time.sleep(random.uniform(3, 6))
        
        # 如果已达到目标数量，提前结束
        if len(all_weibos) >= TARGET_TOTAL:
            print(f'  ✅ 已达到目标数量 {TARGET_TOTAL}，停止抓取')
            break

    # 关键词间等待时间
    if kw_idx < len(KEYWORDS):
        wait_time = random.uniform(15, 25)
        print(f'  等待 {wait_time:.1f} 秒后继续下一个关键词...')
        time.sleep(wait_time)
    
    # 如果已达到目标数量，提前结束所有关键词
    if len(all_weibos) >= TARGET_TOTAL:
        print(f'\n✅ 已达到目标数量 {TARGET_TOTAL}，停止所有抓取')
        break

# 检查是否达到最少数量
if len(all_weibos) < MIN_TOTAL:
    print(f'\n⚠️ 警告：只抓取了 {len(all_weibos)} 条数据，少于目标最少数量 {MIN_TOTAL}')
    print('建议：')
    print('  1. 检查网络连接和Cookie是否有效')
    print('  2. 增加 MAX_PAGES（当前30页）或添加更多关键词（当前已扩展至80+个关键词）')
    print('  3. 检查是否触发反爬虫限制')
    print(f'  4. 如需更多数据，可将 MAX_PAGES 调整至50-100，TARGET_TOTAL 调整至5000+')
else:
    print(f'\n✅ 成功抓取 {len(all_weibos)} 条数据（目标: {MIN_TOTAL}-{TARGET_TOTAL}）')
    if len(all_weibos) >= TARGET_TOTAL:
        print(f'  已达到目标数量，数据充足！')
    elif len(all_weibos) >= MIN_TOTAL:
        print(f'  数据量充足，可进行分析')

# 统计每个关键词的数据量
print('\n📊 各关键词数据统计:')
keyword_stats = {}
for weibo in all_weibos:
    kw = weibo.get('keyword', 'unknown')
    keyword_stats[kw] = keyword_stats.get(kw, 0) + 1

for kw, count in sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True):
    print(f'  {kw}: {count} 条')

output = f'weibo_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output, 'w', encoding='utf-8') as f:
    json.dump(all_weibos, f, ensure_ascii=False, indent=2)

print(f'\n✔ 已保存 {len(all_weibos)} 条微博数据到 {output}')
print(f'   文件大小: {len(json.dumps(all_weibos, ensure_ascii=False).encode("utf-8")) / 1024 / 1024:.2f} MB')
