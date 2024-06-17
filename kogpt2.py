def get_chain(vetor_db, key):   # 벡터DB와 LLM 연결
    llm = ChatOpenAI(openai_api_key=key, model_name='gpt-3.5-turbo', temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetor_db.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return chain   # 벡터DB와 LLM을 연결하는 체인을 리턴


            st.session_state["chain"] = get_chain(vector_db, key)   # 랭체인으로 벡터DB와 GPT LLM을 연결해서 세션에 저장
 




import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os


model_path = os.path.join(os.getcwd(), "models--skt--kogpt2-base-v2", "snapshots", "d0c0df48bf2b2c9350dd855021a5b216f560c0c7")
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained(model_path,
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

question = '건강보험 자격득실에서 중요한 것 뭐냐?'

input_ids = tokenizer.encode(question, return_tensors='pt')
gen_ids = model.generate(input_ids,
                           max_length=128,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)

answer = tokenizer.decode(gen_ids[0])
print(answer)



