import requests

# ！！！请替换成你 settings.py 里填的同一个Cookie字符串！！！
MY_COOKIE = 'WEIBOCN_FROM=1110006030; _T_WM=99895283787; SCF=At_bl9yByv0ENbFBSKWHytS7iH19oSoSfd_9dSXjyskqMABoeCjyLnQJ1gvzU8bXVoijHRwx32Q3KCGyQGa4Du8.; SUB=_2A25EOFU3DeRhGeBM6lUQ-C_Nzz-IHXVnNOj_rDV6PUJbktANLU7ckW1NRDAG61lJHxT9WgTcouUX7_VvbeuFW2Id; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWB.cA_nFF2RAP6OksIX5YY5NHD95Qceo2NeKnpeKB0Ws4Dqcj6i--ciKy2iKysi--fiKysi-8Wi--fi-z7iKysi--4i-zpi-ihi--fiKLhiKnci--fiKLhiKnci--fiKLhiKnc; SSOLoginState=1765549415; ALF=1768141415; MLOGIN=1; XSRF-TOKEN=54918f; mweibo_short_token=34b2f5720a; M_WEIBOCN_PARAMS=oid%3D5243156599145355%26lfid%3D102803%26luicode%3D20000174'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Cookie': MY_COOKIE
}

# 尝试访问需要登录才能完整看到的页面，如个人主页
test_url = 'https://m.weibo.cn/profile/'
# 或者直接用搜索页
# test_url = 'https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3Dtest'

try:
    resp = requests.get(test_url, headers=headers, timeout=10)
    print(f'状态码: {resp.status_code}')
    print(f'响应长度: {len(resp.text)} 字符')
    # 快速检查响应内容是否包含登录信息（如你的昵称关键词）
    if '我的' in resp.text or '未登录' not in resp.text:
        print('>>> Cookie **可能** 有效。')
    else:
        print('>>> Cookie **很可能无效或已过期**，返回了登录页面。')
    # 你可以将部分文本保存下来查看
    with open('test_response.html', 'w', encoding='utf-8') as f:
        f.write(resp.text[:2000]) # 保存前2000字符便于查看
    print('已保存部分响应到 test_response.html，可打开查看。')
except Exception as e:
    print(f'请求出错: {e}')