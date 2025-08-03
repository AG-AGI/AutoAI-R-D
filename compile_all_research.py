import os
import shutil

coder_path = "coder/"

researcher_path = "papers/"

research_path = "research/"

if os.path.exists(research_path):
    shutil.rmtree(research_path)
os.makedirs(research_path)

for source_path in [coder_path, researcher_path]:
    if os.path.exists(source_path):
        for filename in os.listdir(source_path):
            if filename.endswith('.txt'):
                with open(os.path.join(source_path, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                md_filename = filename.replace('.txt', '.md')
                
                with open(os.path.join(research_path, md_filename), 'w', encoding='utf-8') as f:
                    f.write(content)