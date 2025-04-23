import requests
import json
import pandas as pd

# API URL
url = 'https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/O-A0001-001?Authorization=CWA-93D64E8F-55D9-44FE-83BF-260642C8B341&downloadType=WEB&format=JSON'

# 發送請求
response = requests.get(url)
data_json = response.json()

# 確保 API 回傳正確
if 'cwaopendata' in data_json and 'dataset' in data_json['cwaopendata'] and 'Station' in data_json['cwaopendata']['dataset']:
    location_list = data_json['cwaopendata']['dataset']['Station']
    
    # 存放數據的列表
    data_list = []

    # 過濾符合條件的測站
    for station in location_list:
        name = station.get('StationName', '')  # 測站名稱
        station_id = station.get('StationId', '')  # 測站 ID
        obs_time = station.get('ObsTime', {}).get('DateTime', '')  # 觀測時間
        geo_info = station.get('GeoInfo', {})  # 地理資訊
        altitude = geo_info.get('StationAltitude', '')  # 測站高度
        
        # 取得經緯度
        coordinates = geo_info.get('Coordinates', [])
        if coordinates and isinstance(coordinates, list):
            lat = coordinates[0].get('StationLatitude', '')
            lon = coordinates[0].get('StationLongitude', '')
        else:
            lat, lon = '', ''

        # 取得天氣資訊（WeatherElement 內的 Weather 和降水量）
        weather = station.get('WeatherElement', {}).get('Weather', '')  # 天氣狀況
        precipitation = station.get('WeatherElement', {}).get('Now', {}).get('Precipitation', '')  # 即時降水量

        # 確保匹配 "彰化縣 花壇鄉 花壇"
        if geo_info.get('CountyName') == "高雄市" and geo_info.get('TownName') == "大樹區" and name == "大樹":
            data_list.append([name, station_id, obs_time, lat, lon, altitude, weather, precipitation])

    # 存入 DataFrame
    df = pd.DataFrame(data_list, columns=['StationName', 'StationId', 'ObsTime', 'Latitude', 'Longitude', 'Altitude', 'Weather', 'Precipitation'])

    # 儲存為 CSV
    csv_path = 'changhua_huatan_weather.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f'資料已成功儲存到 {csv_path}')
else:
    print("API 資料結構異常或無法取得測站數據")
