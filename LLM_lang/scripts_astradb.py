import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def astradb_start(embeddings=None, collection_name='test'):
    import configparser
    from langchain_community.vectorstores import AstraDB

    config = configparser.ConfigParser()
    config.read('config.ini')
    api_token = config.get('astradb','api_token')
    api_endpoint = config.get('astradb','api_endpoint')

    if embeddings is not None:
        vstore = AstraDB(
            embedding=embeddings,
            collection_name=collection_name,
            api_endpoint=api_endpoint,
            token=api_token,
        )

        return vstore

def astradb_add_pdf(vstore=None, pdf_folder_path='./documents_train/*.pdf'):
    import glob
    from langchain_community.document_loaders import UnstructuredFileLoader
    from unstructured.cleaners.core import clean_non_ascii_chars

    pdf_file_list = []
    for file_name in glob.glob(pdf_folder_path):
        pdf_file_list.append(file_name)

    if vstore is not None and len(pdf_file_list) > 0:
        for pdf_file in pdf_file_list:
            print('Reading {}'.format(pdf_file))
            loader = UnstructuredFileLoader(
                pdf_file,
                extract_images_in_pdf = True,
                chunking_strategy='by_title', 
                max_characters=1024,
                new_after_n_chars=512,
                overlap=256,
                overlap_all=True,
                mode='elements',
                strategy='hi_res',
                # infer_table_structure=True,
                # extract_image_block_types=['Image', 'Table'],
                # extract_image_block_to_payload=True,
                post_processors=[clean_non_ascii_chars],
            )
            docs_from_pdf = loader.load()
            print('Read Complete!')
            print('Documents from PDF: {}.'.format(len(docs_from_pdf)))
            n = 10
            upload_counter = 0
            for docs in [docs_from_pdf[i:i + n] for i in range(0, len(docs_from_pdf), n)]:
                inserted_ids_from_pdf = vstore.add_documents(docs)
                upload_counter += len(inserted_ids_from_pdf)
                print(
                    'Inserted {} out of {} documents.'.format(
                        upload_counter,
                        len(docs_from_pdf)
                    ),
                    end='\r', # https://stackoverflow.com/questions/4897359/output-to-the-same-line-overwriting-previous-output
                )
            print('')
            print('Uploaded {}'.format(pdf_file))
                
def astradb_add_html(vstore=None, html_links_document_path='./link_train.txt'):
    from langchain_community.document_loaders import SeleniumURLLoader

    html_link_list = []
    with open(html_links_document_path, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            html_link_list.append(line.rstrip())

    if vstore is not None and len(html_link_list) > 0:
        loaders = SeleniumURLLoader(
            urls=html_link_list,
            browser='chrome',
            executable_path='chromedriver-mac-arm64/chromedriver'
        )
        docs_from_html = loaders.load()
        print(f"Documents from HTML: {len(docs_from_html)}.")
        n=10
        for docs in [docs_from_html[i:i + n] for i in range(0, len(docs_from_html), n)]:
            inserted_ids_from_pdf = vstore.add_documents(docs)
            print(f"Inserted {len(inserted_ids_from_pdf)} HTML.")


if __name__ == "__main__":
    astradb_start()