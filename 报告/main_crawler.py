import schedule
import time
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)

def daily_crawl():
    """每日定时采集任务"""
    logging.info("开始每日数据采集...")
    
    # 1. 微博数据
    weibo_crawler = WeiboCrawler()
    for keyword in ['星座运势', '塔罗']:
        data = weibo_crawler.search_weibo(keyword, pages=3)
        weibo_crawler.save_data(data, f"{keyword}_{datetime.today().strftime('%Y%m%d')}")
    
    # 2. B站数据
    bili_crawler = BilibiliCrawler()
    videos = bili_crawler.search_videos("玄学", max_pages=2)
    
    logging.info(f"今日采集完成: 微博{len(data)}条, B站{len(videos)}条")

# 设置定时任务（每天凌晨2点）
schedule.every().day.at("02:00").do(daily_crawl)

if __name__ == "__main__":
    # 立即执行一次
    daily_crawl()
    
    # 保持定时任务运行
    while True:
        schedule.run_pending()
        time.sleep(60)