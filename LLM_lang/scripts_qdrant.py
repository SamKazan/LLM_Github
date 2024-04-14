
def qdrant_start(**kwargs):
    import configparser
    from qdrant_client import QdrantClient
    from langchain_community.vectorstores import Qdrant

    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config.get('qdrant','api_key')
    cluster_url = config.get('qdrant','cluster_url')

    embeddings = kwargs.get('embeddings', None)
    collection_name = kwargs.get('collection_name', 'test')

    if embeddings is not None:
        client = QdrantClient(
            url=cluster_url,
            api_key=api_key
        )

        vstore = Qdrant(
            client=client,
            collection_name=collection_name, 
            embeddings=embeddings,
        )

        return vstore

def qdrant_add_pdf(**kwargs):
    import configparser
    import glob
    from langchain_community.document_loaders import UnstructuredFileLoader
    from unstructured.cleaners.core import clean_non_ascii_chars
    from langchain_community.vectorstores import Qdrant

    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config.get('qdrant','api_key')
    cluster_url = config.get('qdrant','cluster_url')

    embeddings = kwargs.get('embeddings', None)
    collection_name = kwargs.get('collection_name', 'test')

    pdf_file_list = []
    for file_name in glob.glob('./documents_train/*.pdf'):
        pdf_file_list.append(file_name)

    if len(pdf_file_list) > 0:
        for pdf_file in pdf_file_list:
            loader = UnstructuredFileLoader(
                pdf_file,
                # extract_images_in_pdf = True,
                chunking_strategy='title', 
                max_characters=512,
                mode='elements',
                post_processors=[clean_non_ascii_chars],
            )
            docs_from_pdf = loader.load()
            print(f"Documents from PDF: {len(docs_from_pdf)}.")
            n=10
            for docs in [docs_from_pdf[i:i + n] for i in range(0, len(docs_from_pdf), n)]:
                inserted_ids_from_pdf = Qdrant.from_documents(
                    docs,
                    embeddings,
                    url=cluster_url,
                    prefer_grpc=True,
                    api_key=api_key,
                    collection_name=collection_name,
                )
                print(f"Inserted {len(inserted_ids_from_pdf)} documents.")

def qdrant_add_html(**kwargs):
    import configparser
    from langchain_community.document_loaders import SeleniumURLLoader
    from langchain_community.vectorstores import Qdrant

    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config.get('qdrant','api_key')
    cluster_url = config.get('qdrant','cluster_url')

    embeddings = kwargs.get('embeddings', None)
    collection_name = kwargs.get('collection_name', 'test')

    html_link_list = []
    with open('./link_train.txt', 'r', encoding='UTF-8') as file:
        while line := file.readline():
            html_link_list.append(line.rstrip())

    if len(html_link_list) > 0:
        loaders = SeleniumURLLoader(
            urls=html_link_list
        )
        docs_from_html = loaders.load()
        print(f"Documents from HTML: {len(docs_from_html)}.")
        n=10
        for docs in [docs_from_html[i:i + n] for i in range(0, len(docs_from_html), n)]:
            inserted_ids_from_pdf = Qdrant.from_documents(
                    docs,
                    embeddings,
                    url=cluster_url,
                    prefer_grpc=True,
                    api_key=api_key,
                    collection_name=collection_name,
                )
            print(f"Inserted {len(inserted_ids_from_pdf)} HTML.")

if __name__ == "__main__":
    qdrant_start()