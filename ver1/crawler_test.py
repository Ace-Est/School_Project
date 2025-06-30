from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import re

# 把這裡換成你的瀏覽器的exe檔位置
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"  

# 這東西我不太確定chrome本身有沒有 
chromedriver_path = "C:/Users/forev/Desktop/chromedriver-win64/chromedriver.exe"

# Set up Brave options
options = webdriver.ChromeOptions()
options.binary_location = brave_path

# Set up driver
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=options)

# Go to the weather page
url="https://www.accuweather.com/en/tw/tauyuan-village/1915899/weather-tomorrow/1915899"
driver.get(url)
#time.sleep(1)  # wait for JS to load

# Extract Precipitation info
try:
    elements = driver.find_elements(By.XPATH, "//p[normalize-space(text())='Precipitation']")
    total_precip = 0.0

    for i, element in enumerate(elements, 1):
        text = element.text
        print(f"[{i}] Precipitation block text:\n{text}")
        match = re.search(r"([0-9]*\.?[0-9]+)\s*mm", text)
        if match:
            value = float(match.group(1))
            print(value)
            total_precip += value
        else:
            print("No number found")

    print(f"\n Total Precipitation: {total_precip:.1f} mm")

except Exception as e:
    print("Could not find Precipitation:", e)

driver.quit()
