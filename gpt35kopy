
import os
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


from langchain.chat_models import ChatOpenAI
import streamlit as st


print(88)

# 현재 스크립트 파일의 디렉토리 경로를 얻기
current_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(current_dir, 'model_cache')  # 'model_cache' 폴더를 캐시 디렉토리로 사용


print(888)
print(current_dir)

print(cache_dir)


model_name = "skt/kogpt2-base-v2"  # 예시 모델 이름, 사용 가능한 최신 모델로 변경 가능


# 모델과 토크나이저를 지정된 캐시 디렉토리에 저장하며 로드
chat_model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)

print(88888)
print(chat_model)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)


print(7)
print(tokenizer)




st.title('건강보험 GPT')

question = st.text_input('GPT 3.5 LLM 기반 챗봇... 질문하기 버튼을 누르세요.')

# 입력 텍스트를 토크나이즈하고 텐서로 변환
input_ids = tokenizer.encode(question, return_tensors='pt')

# 텍스트 생성 설정
generated_outputs = chat_model.generate(
    input_ids,
    max_length=50,  # 최대 길이 설정
    num_return_sequences=1,  # 생성할 문장 수
    no_repeat_ngram_size=2,  # 반복 n-gram 크기를 제한
    temperature=0.7,  # 다양성 조절
    top_k=50,  # 높은 확률을 가진 k개의 토큰만 고려
    top_p=0.95,  # 누적 확률이 p를 넘는 토큰은 무시
    pad_token_id=tokenizer.eos_token_id  # 패딩 토큰 설정
)



if st.button('질문하기'):
    with st.spinner('답변 작성 중...'):
        # result = chat_model.predict(content)
        result = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        print(result)
        st.write(result)






