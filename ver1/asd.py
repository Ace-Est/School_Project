from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import re
import csv

# 路徑設定
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"
chromedriver_path = "C:/Users/forev/Desktop/chromedriver-win64/chromedriver.exe"

options = webdriver.ChromeOptions()
options.binary_location = brave_path

driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
driver.get("https://www.accuweather.com/en/tw/tauyuan-village/1915899/hourly-weather-forecast/1915899")



# 抓所有 hourly 卡片
hourly_cards = driver.find_elements(By.XPATH, "//div[contains(@class, 'accordion-item hour')]")

print(f"🕒 Found {len(hourly_cards)} hourly forecast items.\n")

weather_data = []

for i, block in enumerate(hourly_cards, 1):
    try:
        #展開下拉選單
        if block.get_attribute("data-collapsed") == "true":
            driver.execute_script("arguments[0].click();", block)
            
        # 時間、溫度、降雨機率
        time_str = block.find_element(By.CLASS_NAME, "date").text.strip()
        temp = re.sub(r"[^\d.]", "", block.find_element(By.CLASS_NAME, "temp").text.strip())
        precip = re.sub(r"[^\d]", "", block.find_element(By.CLASS_NAME, "precip").text.strip())

        # 濕度
        try:
            humidity_text = block.find_element(By.XPATH, ".//p[contains(text(), 'Humidity')]/span").text.strip()
            humidity = re.sub(r"[^\d]", "", humidity_text)
        except:
            humidity = ""

        # 雲層覆蓋量
        try:
            Cloud_Cover_text = block.find_element(By.XPATH, ".//p[contains(text(), 'Cloud Cover')]/span").text.strip()
            Cloud_Cover = re.sub(r"[^\d]", "", Cloud_Cover_text)
        except:
            Cloud_Cover = ""

        # 轉換時間為 24 小時
        from datetime import datetime
        hour = datetime.strptime(time_str, "%I %p").hour

        # 固定日期（你可以用 datetime.today().strftime("%Y/%m/%d") 取今天）
        date_str = time.strftime("%Y/%m/%d")

        # 加入清單
        weather_data.append([date_str, temp, humidity, precip, hour, Cloud_Cover])

        print(f"[{i}] Time: {time_str} | Temp: {temp}°C | Precipitation: {precip}% | Humidity: {humidity}%| Cloud_Cover: {Cloud_Cover}%")

    except Exception as e:
        print(f"[{i}] ⚠️ Error extracting data: {e}")

driver.quit()

#  寫入 CSV
with open("hourly_weather.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Temperature", "Humidity", "Rain_Chance", "Hour", "Cloud_Cover"])
    writer.writerows(weather_data)

print("\n✅ Saved to 'hourly_weather.csv'")