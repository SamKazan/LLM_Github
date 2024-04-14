
def huggingface_embed_local(embedding_model):
    from langchain.embeddings import HuggingFaceEmbeddings

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    # model_kwargs = {'device':'cpu'}
    #if using apple m1/m2 -> use device : mps (this will use apple metal)
    model_kwargs = {'device':'mps'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': True}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    print('Testing..')
    query_result = embeddings.embed_query('Some text to test !')
    print(query_result[:3])

    return embeddings

def huggingface_embed_online(embedding_model):
    import configparser
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config.get('huggingface','api_key')

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        # model_name='sentence-transformers/all-MiniLM-l6-v2'
        # model_name='WhereIsAI/UAE-Large-V1'
        model_name=embedding_model
    )

    print('Testing..')
    query_result = embeddings.embed_query('Some text to test !')
    print(query_result[:3])

    return embeddings


if __name__ == "__main__":
    huggingface_embed_online()