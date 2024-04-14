
#HUGGING FACE

from scripts_huggingface import huggingface_embed_local
embedding_model = 'BAAI/bge-large-en-v1.5'
embeddings = huggingface_embed_local(embedding_model)

#ASTRADB

from scripts_astradb import astradb_start
collection_name = "project1"
vstore = astradb_start(embeddings, collection_name)

# PDF to ASTRADB

from scripts_astradb import astradb_add_pdf
astradb_add_pdf(vstore)

#HTML to ASTRADB

from scripts_astradb import astradb_add_html
astradb_add_html(vstore)

#CHAT 
from scripts_chat import chat_start
llm = chat_start()

personality = """
You are a data science professional that draws inspiration from successful data-driven projects of the past
to craft well-thought responses to inquiries.
Your answers must be concise and to the point, and refrain from answering about other topics than data science and analytics.
"""

from scripts_chat import chat_template_1
chain = chat_template_1(llm, vstore, personality)

user_input = '''

 How would you present the information you have as a project that will be pu in github ?
'''
for s in chain.stream(user_input):
    print(s, end="", flush=True)