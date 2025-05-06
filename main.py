from transformers import pipeline

generator = pipeline(task="text-generation", model='deepseek-ai/DeepSeek-Prover-V2-671B')
generator("Hello, I'm a language model", max_length = 30, num_return_sequences=3)



