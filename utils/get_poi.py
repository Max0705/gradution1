from urllib.parse import quote
from urllib import request
import json

ak = "580e03fb6552cc2ea1ad7ccf9356f673"
poi_url = "https://restapi.amap.com/v3/place/polygon?"

max_longitude = '104.589952'
min_longitude = '103.259070'
max_latitude = '31.09167'
min_latitude = '30.17542'

req_url = poi_url + "polygon=" + max_longitude + "|" + max_latitude + "|" + min_longitude + "|" + min_latitude + "&output=json&key=" + ak
with request.urlopen(req_url) as f:
    data = f.read()
    data = data.decode('utf-8')
    data = json.loads(data)
    with open("test.json", "w") as c:
        c.write(json.dumps(data, ensure_ascii=False, indent=4))
