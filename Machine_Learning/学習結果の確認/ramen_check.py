import ramen_keras2 as ramen
import sys, os
from PIL import Image
import numpy as np

# コマンドラインからファイル名を得る --- (※1)
if len(sys.argv) <= 1:
    print("ramen_checker.py (ファイル名)")
    quit()

image_size = 150
categories = ["miso", "sio", "tonkotsu", "shouyu"]


# 入力画像をNumpyに変換 --- (※2)
X = []
files = []
for fname in sys.argv[1:]:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    files.append(fname)
X = np.array(X)

# CNNのモデルを構築 --- (※3)
model = ramen.build_model(X.shape[1:])
model.load_weights("F:/ramen_img100/ramen_model.hdf5")

# データを予測 --- (※4)
html = ""
pre = model.predict(X)
for i, p in enumerate(pre):
    y = p.argmax()
    print("+ 入力:", files[i])
    print("| クラス:", categories[y])

    html += """
        <h3>入力:{0}</h3>
        <div>
          <p><img src="{1}" width=300></p>
          <p>クラス:{2}</p>
        </div>
    """.format(os.path.basename(files[i]),
        files[i],
        categories[y])


# レポートを保存 --- (※5)
html = "<html><body style='text-align:center;'>" + \
    "<style> p { margin:0; padding:0; } </style>" + \
    html + "</body></html>"
with open("ramen-result.html", "w") as f:
    f.write(html)
