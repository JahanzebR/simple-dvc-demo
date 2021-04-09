create env

```bash
conda create -n wineq python=3.7 -y
```

activate env
```bash
conda activate wineq
```

created a req file

install the requirements

```bash
pip install -r requirements.txt
```
download the data from

https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5

git init

dvc init

dvc add data_given/winequality.csv

git add .

git commit -m "first commit"

git remote add origin https://github.com/JahanzebR/simple-dvc-demo.git
git branch -M main
git push -u origin main

onliner updates for readme
git add . && git commit -m "update Readme.md"

git remote add origin https://github.com/JahanzebR/simple-dvc-demo.git
git branch -M main
git push origin main

