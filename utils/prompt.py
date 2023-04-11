import openai

def generate_prompt(prompt_path, input_text):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
        
    return prompt.format(input_text)

def get_chatgpt_answer(prompt, model_name='gpt-3.5-turbo', role='user'):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{
            'role': role, # user/assistant (mean ChatGPT)/system
            'content': prompt
        }]
    )
    
    return completion['choices'][0]['message']['content']

if __name__ == '__main__':
    print(generate_prompt('assets/promts/correction_prompt.txt', 'haha'))