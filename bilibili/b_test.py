import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1️⃣ 启动浏览器
service = Service(r"C:\Users\bdfxl\chromedriver.exe")
options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(service=service, options=options)

kw = "星象分析"
url = f"https://search.bilibili.com/all?keyword={kw}"
driver.get(url)

# 2️⃣ 等待【搜索结果 li】出现（不是等整个页面）
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "li.bili-video-card"))
    )
except:
    print("❌ 搜索结果加载失败")
    driver.quit()
    exit()

# 3️⃣ 滚动，触发异步加载
for _ in range(3):
    driver.execute_script("window.scrollBy(0, 1200)")
    time.sleep(2)

videos = driver.find_elements(By.CSS_SELECTOR, "li.bili-video-card")
print(f"✅ 抓到视频数量：{len(videos)}")

# 4️⃣ 打印前 5 条，确认不是空
for v in videos[:5]:
    title = v.find_element(By.CSS_SELECTOR, "h3").text
    up = v.find_element(By.CSS_SELECTOR, ".bili-video-card__info--author").text
    play = v.find_element(By.CSS_SELECTOR, ".bili-video-card__stats--item").text
    print(title, "|", up, "|", play)

input("按回车关闭浏览器")
driver.quit()
