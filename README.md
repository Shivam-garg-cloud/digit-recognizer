# Handwritten Digit Recognizer

A small machine-learning project that compares three models (K-Nearest Neighbours,
Logistic Regression, Decision Tree) on recognising handwritten digits 0-9.
It uses the dataset built into scikit-learn, so it needs no account, no API key and no internet after installation.

## Run it in VS Code (Mac)
1. Install Python from https://www.python.org/downloads/macos/ and VS Code from https://code.visualstudio.com/
2. In VS Code: File > Open Folder > choose `digit_recognizer`.
3. Open the terminal inside VS Code: Terminal > New Terminal.
4. Run these commands one by one:
       python3 -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt
       python train.py
       pytest
5. Open the new `outputs/` folder: results.csv and three PNG images are your results.
   (If VS Code asks to select a Python interpreter, choose the one inside `venv`.)

## Put it on GitHub
1. Create a free account at https://github.com and click New repository (name: digit-recognizer, Public).
2. In the VS Code terminal:
       git init
       git add .
       git commit -m "Handwritten digit recognizer"
       git branch -M main
       git remote add origin https://github.com/YOUR-USERNAME/digit-recognizer.git
       git push -u origin main
3. On GitHub open the Actions tab: the tests run automatically (green tick = passed).
   Take a screenshot for your report.

## Files
- train.py - loads data, trains and compares the models, saves results and charts
- test_train.py - automated tests
- .github/workflows/ci.yml - runs the tests on GitHub
