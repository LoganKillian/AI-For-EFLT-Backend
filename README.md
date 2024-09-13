# School Achievment Tool
You can check out the full version here [https://ml-school-tool.vercel.app/](https://ml-school-tool.vercel.app/)

Currently an ongoing project of the Dept. of EFLT at Auburn University, this is a Flask and React based machine learning tool to improve Alabama school performance. The models used are run on a static dataset hosted on the flask server. The tool works by first utilizing Lasso Regression to reduce the total amount of features present in the dataset. Then a reverse prediction is made utilizing ExtraTreesRegressor. The user can change the parameters of the models used in each step. 
For each trial, the user is presented with 2 main pieces of data 

1. Lasso provides feature strengths from the whole dataset - this gives a general idea of what broadly impacts performance, positively or negatively
2. Reverse prediction provides the percent change for each feature needed to attain a certain increase in a schools performance
---
### How to Build It Yourself (Current Version)
Flask/backend [here](https://github.com/bls0053/ml-school-flask) and React/frontend [here](https://github.com/bls0053/ml-school-tool).
1. **Clone the Flask Repository**\
    $ `git clone https://github.com/bls0053/ml-school-flask.git`

2. **Activate Python Virtual Environment**\
    $`venv/scripts/activate`  

3. **Install Python Dependencies**\
    $ `pip install -r requirements.txt`

4. **Run the Flask Server**\
    $ `Python main.py`

5. **Clone the React Repository**\
    $ `git clone https://github.com/bls0053/ml-school-tool.git`

6. **Install React Dependencies**\
    $ `npm install`

7. **Run the Development Server**\
    $ `npm run dev`
---
### How to Build It Yourself (Single Repo - outdated version)
###### Note: Monorepo build only available on the 'Final version' commit [here](https://github.com/bls0053/ml-school-tool/commit/16cca6d4d8fcc3ee68e22faeafa2a95af7ab032b) and all previous commits. Later commits split the repo for deployment purposes - Flask/backend [here](https://github.com/bls0053/ml-school-flask) and React/frontend [here](https://github.com/bls0053/ml-school-tool).
1. **Clone the Repository**\
    $`git clone https://github.com/bls0053/ml-school-tool.git`\
    Checkout local version:\
    $`git checkout 16cca6d`

2. **Activate Python Virtual Environment**\
    Navigate into the api directory:\
    $ `cd ml-school-tool/api`\
    Run:\
    $`venv/scripts/activate`

3. **Install Python Dependencies**\
    Navigate into venv:\
    $ `cd ml-school-tool/api/venv`\
    Run:\
    $ `pip install -r requirements.txt`

4. **Run the Flask Server**\
    $ `Flask run`

5. **Install React Dependencies**\
    Navigate into project folder:\
    $ `cd ml-school-tool`\
    Run:\
    $ `npm install`  

6. **Run the Development Server**\
   Start the development server with:\
   $ `npm run dev`

---





