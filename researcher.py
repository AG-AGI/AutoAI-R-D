import json
import os
from model import Ask 
from datetime import datetime

def add_to_research(title, description, filename='research.json'):
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

def read_all_research(filename='research.json'):
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

def save_research_paper(title, content):
    filename = f"{title.replace(' ', '_').replace('/', '_')}.txt"
    filename = "papers/" + filename
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def generate_research_paper():
    print("Generating a new research paper...")
    system_prompt = f'''

    Current Date and Time: {get_current_time()}

    Previous Research Paper Summaries:
    {read_all_research()}

    You are a PhD engineer in the field of computer science, AI R&D field. Your task is to write novel research papers. 
    Ensure that the content is original and contributes to the field.

    Your paper must include a footer at the end with a summary in this format:

    <title_summary>
    The title of the paper
    </title_summary>
    
    <description_summary>
    A brief description of the paper, covering all of the main points. max 2-4 sentences.
    </description_summary>

    Write the research paper with no extra commentary, just send and begin writing the paper directly.

    For your reference/bibliography, cite only a maximum of 3 sources.

    Begin writing a new research paper. 

   '''
    raw_paper = Ask(system_prompt)
    title = raw_paper.split('<title_summary>')[1].split('</title_summary>')[0].strip()
    description = raw_paper.split('<description_summary>')[1].split('</description_summary>')[0].strip()
    
    add_to_research(title, description)
    
    print(f"Generated Paper Title: {title}")
    print(f"Description: {description}\n")

    save_research_paper(title, raw_paper)


Papers_to_generate = 10

for paper in range(Papers_to_generate):
    generate_research_paper()