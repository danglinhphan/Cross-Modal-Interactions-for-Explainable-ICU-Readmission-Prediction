1.EDR
imbalance dataset: LOST + class distribution + ARIMAX
meta weight
ReSpecLoss
Inverse Frequence

1. Machine Learning
2. Deep Learning

WORKFLOW

DATASET
1. Exploratory Data Analysis
    head - shape - describe - info (hiểu thêm về data)
    check integrity data (isnull + sum)
2. Labeling (depend on already data analysis ở trên và tính tương quan của data)
3. Target Distribution
4. Detecting Outliers
5. Merging datasets

MODEL
1. Ensemble (từng model lẻ rồi merge vào 1 model to)
2. train -> reasonable model
3. application section
4. tối ưu model + train thêm 
    a. có thể làm một format, biểu điễn data (làm theo hướng NLP)
    b. đưa data về dạng ma trận
    c. có output + explain 
    d. có list out từng giá trị tại từng epoch để báo cáo (không ngưng train ngay sau khi đặt ngưỡng + thêm 10 epoch dư)
    e. lưu ý cần chuẩn hoá các chỉ số đánh giá model (Recall, F1, Precision, Accuracy)
5. explainable

evaluation
only structure data 
only texture data
structure + texture
structure + texture + pair

validate
example to show eplaination 
LLMs -> XAI 
traditional method on different feature (novoty feature)
how to represent patient 
why I do it, Why it work
retrieve the relevant sentence 

1. literature review for LLMs
2. explaination summary
3. NLP friendly, easy to understand
4. from this feature generate NLP summary
5. explain the reason why the addition of the pair can improve accuracy.


compare predciton
other machine learning method 

# Terminal 1 - Backend
cd /Users/phandanglinh/Desktop/VRES
source .venv/bin/activate
uvicorn LLM_Explanation.api:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend  
cd /Users/phandanglinh/Desktop/VRES/ebm-llm-frontend
pnpm run dev