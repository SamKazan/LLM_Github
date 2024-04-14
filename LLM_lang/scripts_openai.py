
def openai_embed_local():
    import configparser
    from langchain_openai import OpenAIEmbeddings

    config = configparser.ConfigParser()
    config.read('config.ini')
    base_url = config.get('openai','base_url')
    api_key = config.get('openai','api_key')
    model_name = config.get('openai','model_name')

    api_config = {
        'base_url': base_url,
        'api_key': api_key,
        'model': model_name,
        # 'streaming': True,
        'dimensions': 1024
    }

    embeddings = OpenAIEmbeddings(**api_config)

    print('Testing..')
    query_result = embeddings.embed_query('Some text to test !')
    print(query_result[:3])

    return embeddings


if __name__ == "__main__":
    openai_embed_local()