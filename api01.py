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






