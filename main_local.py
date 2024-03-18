from transformers import AutoTokenizer, AutoModelForCausalLM


"""
import os
import streamlit as st

# 현재 스크립트 파일의 디렉토리 경로를 얻기
current_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(current_dir, 'model_cache')  # 'model_cache' 폴더를 캐시 디렉토리로 사용

print(current_dir)
print(cache_dir)

model_name = "skt/kogpt2-base-v2"  # 예시 모델 이름, 사용 가능한 최신 모델로 변경 가능

@st.cache_data
def load_model_and_tokenizer(model_name, cache_dir):
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer
"""






# 저장된 모델 경로 설정
model_path = "D:/Project/gemma2b/models--beomi--gemma-ko-7b/snapshots/9c24b9c4ab362ca141e8dc1c8fb9cb124c1a136e"

print("start")
# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("tokenizer 완료")

model = AutoModelForCausalLM.from_pretrained(model_path)
print("model 로드 완료")

# 모델 사용 예시
input_text = "안녕하세요, 이 모델은 한글 텍스트 생성에 사용됩니다."
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
outputs = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=3, do_sample=True)

for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))