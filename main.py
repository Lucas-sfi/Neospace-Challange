from openai import OpenAI
from datasets import Dataset, DatasetDict, load_dataset
import json


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

topic = "Tópicos gerais"

n_subtopics = 200
n_questions = 200

TOPIC_GENERATION_PROMPT_TEMPLATE = "sobre um topico crie uma lista de {n_subtopics} subtopicos, que estão relacionados com o tópico, o tópico é {topic}, a lista nao deve ter números, e nao deve ter descrições dos subtopicos, apenas os subtopicos"

def generate_subtopics(client, topic, n_subtopics):
    prompt = TOPIC_GENERATION_PROMPT_TEMPLATE.format(topic=topic, n_subtopics=n_subtopics)
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response

responses = generate_subtopics(client, topic=topic, n_subtopics=n_subtopics)
print(responses.choices[0].message.content)

QUESTION_PROMPT_TEMPLATE = " sobre um certo topico, crie {n_questions} perguntas que podem ser perguntadas sobre esse tópico, a resposta deve estar no formato de uma lista, o topico é {sub_topic}, a lista não deve ter números. As perguntas devem estar separadas por um caracter de nova linha. Nao deve existir nada além do texto presente na lista, nem todas as " \
"perguntas estarão no modelo: qual a capital da frança?, algumas devem estar em formato de ordem: me dê a capital da frança"

subtopic_list = responses.choices[0].message.content.split(",")
def generate_questions(client, sub_topic, n_questions):
    prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def question_generator(client, subtopic_list, n_question):
    tasks = [generate_questions(client, subtopic, n_question) for subtopic in subtopic_list]
    question_list = tasks
    return question_list

question_list = question_generator(client, subtopic_list, n_questions)
print(question_list)

question_list_formatted = []
for question_set in question_list:
    question_list_formatted.extend([question.strip() for question in question_set.split("\n") if question])
len(question_list_formatted)

RESPONSE_PROMPT_TEMPLATE = """\
Dado uma certa pergunta, gere apenas 1 resposta que podem responder essa pergunta, sua reposta deve ter tom, gramática e léxica parecida com o personagem de ficção Chico Bento da Turma da Mônica, responda com todas as informações pertinentes, mude apenas o modo de escrevê-las e as escreva como se fosse o Chico Bento, usando seu dialeto capiria do interior de São Paulo.

Exemplos de fala de Chico Bento:

 "O cê tá querendo que eu tire a arvre daquele canto?"
 (Você está querendo que eu tire a árvore daquele canto?) - Exemplo de "ocê" e da pronúncia do "r". 

"Ó, as pranta tá crescendo, moço!"
(Olha, as plantas estão crescendo, rapaz!) - Exemplo da pronúncia do "r" e do uso de "pranta" em vez de "planta". 
"Eu vou lá e toco lá pras tias."
(Eu vou lá e toco lá para as tias) - Exemplo de "toquei" no lugar de "tocou" e do uso de "tias" em vez de "tias". 
"Deve de sê bom"
(Deve ser bom) - Exemplo de apócope, onde a letra "r" é omitida no final da palavra. 
"Eu vi uma viage pro Sampa."
(Eu vi uma viagem para São Paulo) - Exemplo da redução da nasal final (viagem para viagem). 
"Qui que eu faço?"
(O que que eu faço?) - Exemplo de "qui" em vez de "que". 
 Sua pergunta deve estar em formado de lista.

A pergunta é: {question}

A lista deve estar nesse formato:

"testo da resposta"

Nada além da resposta devem estar nas respostas, apenas:
"texto da resposta"
"""
def generate_responses(client, question):
    prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def response_generator(client, question_list):
    tasks = [generate_responses(client, question) for question in question_list]
    response_list = tasks
    return response_list

question_response_list = response_generator(client, question_list_formatted)
question_response_pair_list = []
for question, response_set in zip(question_list_formatted, question_response_list):
    question_response_pair_list.append(
        {
            "question": question,
            "responses": response_set
              
        }
    )

    with open('synthetic_data.jsonl', 'w') as f:
     for item in question_response_pair_list:
        f.write(json.dumps(item))
        f.write('\n') 

#    with open('synthetic_data.jsonl', 'r') as f:
 #    data = [json.loads(line) for line in f]
  #  dataset = Dataset.from_list(data)
   # dataset_dict = DatasetDict({"train": dataset})
    #dataset_dict.push_to_hub("Lucas-sf/chico-bento-dataset-test")

    





