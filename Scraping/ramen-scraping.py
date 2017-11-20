# coding: utf-8

from bs4 import BeautifulSoup
import time, re , os
import urllib.request as req
import random
import csv
import requests as reqs
ID = 1 #IDを関数の外で宣言

def sel_one(writer, dat):
    if dat.select_one("div.body > div.comment > p > a > img") != None:# reviews have image
        score = dat.select_one(".score").text
        score = score.replace("点", "")#点をとる
        style = dat.select_one("div.title > div.menu > span").text
        shop = dat.select_one("span > a").text
        comment = dat.select_one("div.body > div.comment > p > span").text
        comment = comment.replace("\n", "")#改行文字を置換
        img_url = dat.select_one("div.body > div.comment > p > a > img").get("src")

        data_one = []
        global ID
        data_one.append(ID)#レビュー毎にＩＤを振る
        data_one.append(score)
        data_one.append(style)
        data_one.append(shop)
        data_one.append(comment)
        data_one.append(img_url)
        store_one(writer, data_one)
        ID += 1

def get_img(dat):
    os.chdir('F:/ramen_image')# 画像は別ディレクトリに
    ID = dat[:][0]#リストのはじめからＩＤを取得
    ID = str(ID) + ".jpg"#ＩＤを画像のファイル名に
    img_url = dat[:][-1]
    r = reqs.get(img_url)
    bin = r.content

    with open(ID, mode= "ab")as f:
        f.write(bin)
        print(ID)
        print("保存しました")


def sel(writer, shop_rev):# select 1shop review data
	for items in shop_rev:
         sel_one(writer, items)


def store_one(writer, dat):#store data
	writer.writerow(dat)
	get_img(dat)

def finish(f):
	f.close

def  scrape():
    # Setup
    header = ['ID', 'score','style','shop', 'comment','img']

    f = open('ramen.csv', 'w', newline = "", encoding='UTF=8')#空行を無くす
    writer = csv.writer(f)
    writer.writerow(header)

    page = 1
    while page > 0:
        try:
            url ="https://ramendb.supleks.jp/reviews?page={0}".format(page)
            res = req.urlopen(url)
            soup = BeautifulSoup(res, "html.parser")
            shop_rev = soup.select(".review-shop.border-box")
            sel(writer, shop_rev)# 引数にファイルオブジェクトとデータ
            page += 1
            sleep = random.randint(1,10)
            time.sleep(sleep) # sleep 1~10sec
            print(page)

        except:
            finish(f)
            print("ダウンロード完了")
            break

scrape()
