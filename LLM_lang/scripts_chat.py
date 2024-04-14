import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def chat_start():
    import configparser
    from langchain_openai import ChatOpenAI
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    base_url = config.get('openai','base_url')
    api_key = config.get('openai','api_key')
    model_name = config.get('openai','model_name')

    api_config = {
        'base_url': base_url,
        'api_key': api_key,
        'model_name': model_name,
        'streaming': True,
    }

    llm = ChatOpenAI(**api_config)

    return llm

def chat_template_1(llm=None, vstore=None, personality=None):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    if llm is not None and vstore is not None:

        retriever = vstore.as_retriever(search_kwargs={'k': 3})


        philo_template = """
        {personality}
        
        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:

        """

        philo_prompt = ChatPromptTemplate.from_template(philo_template)

        chain = (
            {'personality': lambda x: personality, 'context': retriever, 'question': RunnablePassthrough()}
            | philo_prompt
            | llm
            | StrOutputParser()
        )

        return chain



if __name__ == "__main__":
    chat_start()