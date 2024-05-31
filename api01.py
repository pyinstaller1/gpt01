import requests

url = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
params ={'serviceKey' : '서비스키', 'YM' : '201201', 'NAT_CD' : '112', 'ED_CD' : 'E' }

response = requests.get(url, params=params)
print(response.content)






import json
import urllib.request

key = "키"
URL = "hrrp://www/key=" + key + "&data="
json_page = urllib.request.urlopen(URL)
json_data = json_page.read().decode("utf-8")
json_array = json.loads(json_data)

print(json_array)


import requests
from datetime import datetime








url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'
params ={'serviceKey' : 'LeXut7UmcYum2fOusIxBEeUSjTyABxPbM2odYVlwkRYAWVCbsT/xx4bUIO52VxqMUO/yZUT60MwFoCdbQLr1sw==', 'pageNo' : '1', 'numOfRows' : '1000', 'dataType' : 'JSON', 'base_date' : datetime.now().strftime('%Y%m%d'), 'base_time' : datetime.now().strftime('%H%M'), 'nx' : '77', 'ny' : '122' }

response = requests.get(url, params=params)

print(response.content)
print(datetime.now().strftime('%H%M'))

response_json = response.json()

for item in response_json["body"]["items"]["item"]:
    if item["category"] == "T1H":
        print("현재 기온:", item["obsrValue"], "°C")
        break




