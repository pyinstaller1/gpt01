from urllib.parse import urlencode
import urllib.request
import json
import pandas as pd
from pandas import json_normalize
from datetime import datetime



def getbiz(maplist):
    values ={'serviceKey' : maplist['sKey'], 'numOfRows' : '900', 'pageNo' : '1', 'inqryDiv' : '1', 'inqryBgnDt' : maplist['StartDt'], 'inqryEndDt' : maplist['EndDt'],
            'indstrytyCd' : maplist['Cd'], 'type' : 'json' }
    param = urlencode(values)
  
    # 요청 URL에 Parameter를 붙여 생성
    url = maplist['API'] + "?" + param
    #response = urllib.request.urlopen(url).read().decode('utf-8')
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    
    # 공공데이터 요청에 대한 정상 응답여부 확인
    if (rescode==200):
        response_body = response.read().decode('utf-8')
    else:
        print("Error Code : "+rescode)
        return(rescode)
    
    # 문자열을 JSON으로 변환 작업, 닥션너리형태로 반환 { Key:Value }
    json_object = json.loads(response_body)
    #print(json_object)

    # DataFrame 로 변환
    result = json_normalize(json_object['response']['body']['items'])
    #print(result.head(1))
    
    # Vectorization을 이용하여 문자열 컬럼 분할 : 낙찰업체 주소에서 시와 구를 분리하여 별도 컬럼에 추가 한다
    # bitwinnrAdrs : 낙찰업체주소 항목명
    # str.split() : 문자열을 구분자로 분리
    result['country'] = result.bidwinnrAdrs.str.split(' ').str[0]
    result['gu'] = result.bidwinnrAdrs.str.split(' ').str[1]

    # 엑셀로 출력
    result.columns = ['입찰공고번호', '입찰공고차수', '입찰분류번호', '재입찰번호', '공고구분코드', '입찰공고명', 
                      '참가업체수', '최종낙찰업체명', '최종낙찰업체사업자등록번호', '최종낙찰업체대표자명','최종낙찰업체주소',
                      '최종낙찰업체전화번호','최종낙찰금액', '최종낙찰률','실개잘일시','수요기관코드','수요기관명', '등록일시', '최종낙찰일자',
                      '최종낙찰업체담당자', '연계기관명', '시', '구']
    
    try:
        #날짜 받아오기 : datetime.now() 현재날짜를 가져온다
        # strftime : 날짜 타입을 string 타입으로 변경
        filename_dt = datetime.now().strftime("%Y-%m-%d")
        #파일명 prefix 지정 후 현재날짜를 붙여 엑셀로 저장, index=false -> Index(1,2,3,..)표시를 엑셀에서는 표시하지 않게함
        result.to_excel(r'c:\My files\work\ch02\output\낙찰업체정보-'+filename_dt+'.xlsx', sheet_name='Sheet1', index=False)        
        return(0)
    except:
        print("예외가 발생했습니다.")    
        return(100)
    
def main():    
    inputValues = {'API' : 'http://apis.data.go.kr/1230000/ScsbidInfoService/getScsbidListSttusServcPPSSrch',
                   'sKey' : '인증키입력',
                   'StartDt' : '202201010000',
                   'EndDt' : '202212310000',
                   'Cd' : '6146'
                   }    
    # sKey : 개별 신청한 인증키 입력, StartDt : 조회 시작일, EndDt : 조회 종료일,
    # Cd : 업종코드(※ 6146 : 정보시스템 감리법인)
    installProxy()
    result = getbiz(inputValues)
    if (result == 0):
        print("정보조회가 완료했습니다.")
    else:
        print("정보조회가 비정상 종료했습니다.")
    
if __name__ == "__main__":
    main()





























from urllib.parse import urlencode
import urllib.request
import pandas as pd
from datetime import datetime

def installProxy():
    try:
        proxy = urllib.request.ProxyHandler({'http':'proxy ip 주소:port', 'proxy ip 주소:port'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
        print("Open Proxy 성공")
    except:
        print("Proxy 설정에 오류가 발생했습니다.")

def getbiz(maplist):
    values ={'serviceKey' : maplist['sKey'], 'LAWD_CD' : maplist['LAWD_CD'], 'DEAL_YMD' : maplist['DEAL_YMD'] }
    api = maplist['API']
    
    # Land_CD : 11740 -> 강동구
    # urlencode : 파라메터로 변경, 이때 Key는 일반decode된 키를 사용한다.
    param = urlencode(values)
    print(param)
 
   # 요청 URL에 파라메터를 붙여 만든다.
    url = maplist['API'] + "?" + param
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    
    # 공공데이터 요청에 대한 정상 응답여부 확인
    if (rescode==200):
        response_body = response.read().decode('utf-8')
        bs_obj = bs(response_body, "lxml-xml")
    else:
        print("Error Code : "+rescode)
        return(rescode)
    # 필요한 정보는 'item'에 있으므로 'item' 태그 정보를 모두 가져 온다.    
    rows = bs_obj.findAll('item')

    # 조회결과를 저장 할 list 타입의 변수를 생성
    rowList = []
    nameList = []
    columnList = []
    
    # 전체 Row수를 저장
    rowsLen = len(rows)

    for i in range(0, rowsLen):
        columns = rows[i].find_all()
        columnsLen = len(columns)
    
        for j in range(0, columnsLen):
            # 항목명을 첫번째 Row에 저장
            if i == 0:
                nameList.append(columns[j].name)
            # 각 컬럼값을 list에 저장
            eachColum = columns[j].text
            columnList.append(eachColum)
        rowList.append(columnList)
        # 컬럼list를 초기화
        columnList = []

    try:
        filename_dt = datetime.now().strftime("%Y-%m-%d")
        result = pd.DataFrame(rowList, columns=nameList)
        result.to_excel(r'c:\My files\10.Python\work\ch02\output\아파트전월세현황_'+filename_dt+'.xlsx', sheet_name='Sheet1', index=False)
    except:
        print('예외가 발생했습니다.')

def main():
    inputValues = {'API' : 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent',
                 'sKey' : 'decoding된 인증키 입력',
                 'LAWD_CD' : '11740', 'DEAL_YMD' : '202203'
                }    
    installProxy()
    ret = getbiz(inputValues)
    
if __name__ == "__main__":
    main()












