import json
import os
from model import Ask 
from datetime import datetime

path = "coder"

def add_to__code_research(title, description, filename=f'{path}.json'):
    entry = {"title": title, "description": description}
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError
            except:
                data = []
    else:
        data = []
    data.append(entry)
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def read_all__code_research(filename=f'{path}.json'):
    if not os.path.exists(filename):
        return "[]"
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            if not isinstance(data, list):
                raise ValueError
            return json.dumps(data, indent=2, ensure_ascii=False)
        except:
            return "[]"

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_code_research_paper(title, content):
    filename = f"{title.replace(' ', '_').replace('/', '_')}.txt"
    filename = "coder/" + filename
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def generate_code_research_paper():
    print("Programmer is coding....")
    system_prompt = f'''

    Current Date and Time: {get_current_time()}

    Previous Research Paper Summaries:
    {read_all__code_research()}

    You are a PhD programmer in the field of computer science, AI R&D field. Your task is to write novel research papers and code. 
    Ensure that the content is original and contributes to the field.

    Your paper must include a footer at the end with a summary in this format:

    <title_summary>
    The title of the paper
    </title_summary>
    
    <description_summary>
    A brief description of the paper, covering all of the main points. max 2-4 sentences. including any relevant code snippets.
    </description_summary>

    <paper_main_code>
    Write code in python that showcases the main points of the paper.
    </paper_main_code>

    Write the research paper with no extra commentary, just send and begin writing the paper directly.

    For your reference/bibliography, cite only a maximum of 3 sources.

    make sure your paper uses markdown and multiple code examples, from python, javascript, c++, java, or any other programming language you decide to use in this paper.

    Your main goal is to research AI architectures, create new ones, improve them, NLP, ML, LLMs, softmax, gradient descent, and any other AI related topic. and produce relevant code in the paper and paper summary.

    Begin writing a new research paper. 

   '''
    raw_paper = Ask(system_prompt)
    title = raw_paper.split('<title_summary>')[1].split('</title_summary>')[0].strip()
    description = raw_paper.split('<description_summary>')[1].split('</description_summary>')[0].strip()

    paper_main_code = raw_paper.split('<paper_main_code>')[1].split('</paper_main_code>')[0].strip().replace("```python", "").replace("```", "").replace("```", "").strip()

    code_filename = f"coder/{title.replace(' ', '_').replace('/', '_')}.py"
    os.makedirs(os.path.dirname(code_filename), exist_ok=True)
    with open(code_filename, 'w', encoding='utf-8') as file:
        file.write(paper_main_code)

    add_to__code_research(title, description)

    print(f"Generated Paper Title: {title}")
    print(f"Description: {description}\n")

    save_code_research_paper(title, raw_paper)


if __name__ == "__main__":
    for _ in range(10):
        generate_code_research_paper()