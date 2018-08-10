import os
import json
import sys
import re
import model
import decoder

homepath = os.path.abspath(os.path.dirname(__file__)).rsplit("/",1)[0]
exec_path = homepath
conf_path = exec_path+"/common/config.json"

def load_user_data(fileName):
    #ファイルを開いて、データを読み込んで変換する
    #データ形式は(user,password)
    try:
        file = open(fileName,'r')
        a = json.loads(file.read())
        file.close()
    except:
        sys.exit(1)
    return a


def parse_text(t):
    result = re.sub("@[a-zA-Z_0-9]+", "", t)
    result = re.sub("https?://t co/[/a-zA-Z_0-9]+", "", result)
    result = re.sub("https?:[/a-zA-Z_0-9]+", "", result)
    result = re.sub("\r\n|\n|\r", "", result)
    return result 


def analyze_generate():
    a = load_user_data(conf_path)
    dbSession = model.startSession(a)
    dec = decoder.Decoder()    

    q = dbSession.query(model.Tweet)
    tq = q.filter(model.Tweet.isAnalyze < 2)[:10000]
    for t in tq:
        print(t.user, parse_text(t.text))
        print(dec.decode(parse_text(t.text)))
        

if __name__ == '__main__':
    analyze_generate()
