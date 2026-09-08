# OA_detection_system
Osteratrhiritis detection system with Questionnaire and x-ray verification through especially trained AI model
python3 -m venv oa_env
source oa_env/bin/activate
python3 tabular_model.py
streamlit run frontend.py
CHECKPOINT_PATH="./checkpoints/oa_resnet18_binary.pt" uvicorn app:app --reload --port 8000 &
curl http://localhost:8000/health


## Quickstart Instructions

### 1. Repository Setup & Dependencies
```bash
git clone [https://github.com/bismuth-frogs/OA_detection_system.git](https://github.com/bismuth-frogs/OA_detection_system.git)
cd OA_detection_system

python3 -m venv oa_env
source oa_env/bin/activate


mkdir -p data/raw

curl -L -H "User-Agent: Mozilla/5.0" -o data/raw/DEMO_J.XPT [https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT)
curl -L -H "User-Agent: Mozilla/5.0" -o data/raw/BMX_J.XPT [https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT)
curl -L -H "User-Agent: Mozilla/5.0" -o data/raw/MCQ_J.XPT [https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/MCQ_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/MCQ_J.XPT)
curl -L -H "User-Agent: Mozilla/5.0" -o data/raw/PAQ_J.XPT [https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT)
curl -L -H "User-Agent: Mozilla/5.0" -o data/raw/OCQ_J.XPT [https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OCQ_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OCQ_J.XPT)


