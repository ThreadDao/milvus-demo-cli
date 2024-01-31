import sys
import readline
import random
import string
import click
import pyarrow.parquet as pq
import numpy as np
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType, MilvusException

columns = ["pk", "caption", "NSFW", "width", "height", "float32_vector"]
output_fields = ["pk", "caption", "NSFW", "width", "height"]
data_path = 'data/laion1b_nolang_768d_1w.parquet'
query_df = pq.read_table("data/query.parquet", columns=columns).to_pandas()
query_vec = list(query_df["float32_vector"])

"""
util func
"""


def print_help_msg(command):
    with click.Context(command) as ctx:
        click.echo(command.get_help(ctx))


def get_recall_value(true_ids, result_ids):
    intersection = set(true_ids).intersection(set(result_ids))
    return round(len(intersection) / len(result_ids), 3)


"""
click group and command
"""


@click.group()
def milvus():
    pass


@milvus.command()
@click.option('--host', default="localhost", help='host of milvus')
@click.option('--port', default=19530, help='port of milvus')
def connect(host, port):
    """Connect to milvus."""
    global conn
    conn = connections.connect(host=host, port=port)


@milvus.command()
def disconnect():
    """Disonnect to milvus."""
    connections.disconnect("default")


@milvus.command()
def list_collections():
    """ List all collections"""
    click.echo(f"Collections: {utility.list_collections()}")


@milvus.command()
def clear_collections():
    """ Drop all collections"""
    click.echo('Start to clear all collections...')
    click.echo(f"Collections: {utility.list_collections()}")
    for c in utility.list_collections():
        utility.drop_collection(c)
    assert utility.list_collections() == []


@milvus.command("exit")
def quit_cli():
    """Exit the CLI"""
    global quit_cli
    quit_cli = True


@milvus.command()
def help():
    """Show help messages"""
    click.echo(print_help_msg(milvus))


@milvus.command()
def show_data():
    """Show parts data of laion1b-nolang 768dim"""
    click.echo('Start to show data...')
    if not os.path.isfile(data_path):
        click.echo(f"File {data_path} not exist or not a file, please check!")
    if data_path.endswith("parquet"):
        data_df = pq.read_table(data_path, columns=columns).to_pandas()
        click.echo("The dtypes of dataframe is:")
        click.echo(data_df.dtypes)
        click.echo(data_df[:10])
    else:
        click.echo("Only parquet file is supported now.")


@milvus.command()
def prepare_collection():
    """Prepare a random name collection: create -> insert -> flush -> index -> load"""

    # prepare field schema
    click.echo(click.style('Prepare schema of field and collection...', fg='green'))
    pk_id_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True)
    caption_field = FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=2048)
    NSFW_field = FieldSchema(name="NSFW", dtype=DataType.VARCHAR, max_length=2048)
    width_field = FieldSchema(name="width", dtype=DataType.INT64)
    height_field = FieldSchema(name="height", dtype=DataType.INT64)
    float32_vector_field = FieldSchema(name="float32_vector", dtype=DataType.FLOAT_VECTOR, dim=768)

    # prepare collection schema
    schema = CollectionSchema(
        fields=[pk_id_field, caption_field, NSFW_field, width_field, height_field,
                float32_vector_field],
        description="Demo schema from laion1b_nolang dataset.",
    )

    # create collection
    click.echo(click.style("Prepare collection and data...", fg="green"))
    collection_name = "demo_" + "".join(random.choice(string.ascii_letters) for _ in range(4))
    collection = Collection(name=collection_name, schema=schema)
    collections = utility.list_collections()
    assert collection.name in collections
    click.echo(click.style(f"Collection: {collection.name} created successfully!", fg="green"))

    # insert data
    df = pq.read_table(data_path, columns=columns).to_pandas()
    collection.insert(df)
    collection.flush()
    click.echo(click.style(f"Insert entities {collection.num_entities} successfully", fg="green"))

    click.echo(click.style(f"Prepare index and load...", fg="green"))
    # create index
    index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 30, "efConstruction": 360}}
    collection.create_index("float32_vector", index_params=index_params, _async=False)

    # load
    collection.load()
    click.echo(click.style(f"Prepare collection {collection.name} Done!", fg="green"))


# search with limit
@milvus.command()
@click.option('--collection_name', default="collection_name", help='collection name to range search')
def search(collection_name):
    """ Search collection with limit 100"""

    collection = Collection(name=collection_name)
    # search
    param = {"metric_type": "COSINE", "params": {"ef": 100}}
    click.echo(click.style("*"*30 + "Query entity is:", fg="blue"))
    click.echo(query_df)
    click.echo(click.style("*"*30 + "Search and returns 100 results" + "*" * 30, fg="green"))
    search_res = collection.search(data=query_vec, anns_field='float32_vector', param=param, limit=100,
                                   output_fields=output_fields)

    click.echo(search_res[0][0])
    click.echo(search_res[0][1])
    click.echo(click.style("                   .", fg="blue"))
    click.echo(click.style("                   .", fg="blue"))
    click.echo(click.style("                   .", fg="blue"))
    click.echo(search_res[0][-2])
    click.echo(search_res[0][-1])
    true_ids = np.load("data/ids_limit_100.npy")
    click.echo(
        click.style(f"***************************** Recall is {get_recall_value(true_ids.tolist(), search_res[0].ids)} "
                    f"*****************************", fg="red"))


# range search
@milvus.command()
@click.option('--collection_name', default="collection_name", help='collection name to range search')
def range_search(collection_name):
    """ Search collection with limit 100 and distance 0.75 ~ 1.0"""
    # init collection
    collection = Collection(name=collection_name)
    param = {
        "metric_type": "COSINE",
        "params": {"ef": 100, "radius": 0.7, "range_filter": 1.0}
    }
    click.echo(click.style("*"*30 + "Query entity is:", fg="blue"))
    click.echo(query_df)
    click.echo(click.style("********** Search distances ranging from 0.75 to 1.0 and returns 100 results. **********", fg="green"))

    search_res = collection.search(data=query_vec, anns_field='float32_vector', param=param, limit=100,
                                   output_fields=output_fields)
    click.echo(search_res[0][0])
    click.echo(search_res[0][1])
    click.echo(click.style("                   .", fg="blue"))
    click.echo(click.style("                   .", fg="blue"))
    click.echo(click.style("                   .", fg="blue"))
    click.echo(search_res[0][-2])
    click.echo(search_res[0][-1])
    true_ids = np.load("data/ids_range_0.75_1.npy")
    click.echo(
        click.style(f"********************* Recall is {get_recall_value(true_ids.tolist(), search_res[0].ids)} *"
                    f"*********************", fg="red"))


# range search with expr filter
@milvus.command()
@click.option('--collection_name', default="collection_name", help='collection name to range search')
def filter_search(collection_name):
    """ Search collection with limit 100 and distance 0.75 ~ 1.0, and expr: 'width < 500' """
    collection = Collection(name=collection_name)
    # search filter
    param = {
        "metric_type": "COSINE",
        "params": {"ef": 100, "radius": 0.7, "range_filter": 1.0}
    }
    click.echo(click.style("*"*30 + "Query entity is:", fg="blue"))
    click.echo(query_df)
    click.echo(
        click.style("********** Search with expr: width < 500 and distances ranging from 0.75 to 1.0 and returns 100 results. ***"
                    "*******", fg="green"))
    search_res = collection.search(data=query_vec, expr='width < 500', anns_field='float32_vector', param=param,
                                   limit=100, output_fields=output_fields)
    click.echo(search_res[0][0])
    click.echo(search_res[0][1])
    click.echo(click.style("                   .", fg="blue"))
    click.echo(click.style("                   .", fg="blue"))
    click.echo(click.style("                   .", fg="blue"))
    click.echo(search_res[0][-2])
    click.echo(search_res[0][-1])
    true_ids = np.load("data/ids_range_expr.npy")
    click.echo(
        click.style(f"************************ Recall is {get_recall_value(true_ids.tolist(), search_res[0].ids)} **************"
                    f"********", fg="red"))


@milvus.command()
@click.option('--collection_name', default="collection_name", help='collection name to range search')
def iterator_search(collection_name):
    """ Iterator search collection with limit 100 and distance 0.75 ~ 1.0, expr: 'width < 500', and batch_size 10 """
    click.echo('Start to iterator search...')
    collection = Collection(name=collection_name)

    # search iterator
    param = {"metric_type": "COSINE", "params": {"ef": 100, "radius": 0.7, "range_filter": 1.0}}
    click.echo(click.style("*"*30 + "Query entity is:", fg="blue"))
    click.echo(query_df)
    click.echo(
        click.style("********** Iterative search with expr: 'width < 500' and returns 100 results in batches of 10. ****"
                    "******", fg="green"))
    search_iterator = collection.search_iterator(data=query_vec, expr='width < 500', batch_size=10, limit=100,
                                                 anns_field='float32_vector', param=param, output_fields=output_fields)
    result_ids = []
    true_ids = np.load("data/ids_expr_iterator.npy")
    page = 0
    while True:
        res = search_iterator.next()
        if len(res) == 0:
            click.echo(click.style("********** Search iteration finished, close. **********", fg="green"))
            search_iterator.close()
            break
        else:
            page += 1
            click.echo(click.style(f"********************************** Result of page {page} *****************"
                                   f"*****************", fg="red"))
        for i in range(len(res)):
            result_ids.append(res[i].id)
        click.echo(res[0])
        click.echo(click.style(f"--------------------------------------- *{len(res) - 2}-----------------------"
                               f"----------------------", fg="white"))
        click.echo(res[-1])
        click.pause()
    click.echo(
        click.style(f"**************************** Recall is {get_recall_value(true_ids, result_ids)} *********"
                    f"*************", fg="red"))


import os


def get_colors(ctx, args, incomplete):
    colors = [('red', 'help string for the color red'),
              ('blue', 'help string for the color blue'),
              ('green', 'help string for the color green')]
    return [c for c in colors if incomplete in c[0]]


quit_cli = False  # global flag


def runCliPrompt():
    try:
        while not quit_cli:
            readline.set_completer_delims(" \t\n;")
            readline.parse_and_bind("tab: complete")
            astr = input("milvus_demo > ")
            try:
                milvus(astr.split())
            except SystemExit:
                continue
            except MilvusException as ce:
                click.echo(
                    message=f"Milvus Error {str(ce)}!\nPlease check!.",
                    err=True,
                )
            except Exception as e:
                click.echo(message=f"Error occurred!\n{str(e)}", err=True)
        print("Bye~")
    except (KeyboardInterrupt, EOFError):
        print("Bye~")
        sys.exit(0)


if __name__ == '__main__':
    runCliPrompt()
    # 10.102.7.29
    # demo_bmZJ
    """
    (milvus-demo) ➜  milvus-demo python3 main.py
    milvus_demo > help
    Usage:  [OPTIONS] COMMAND [ARGS]...
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      clear-collections   Drop all collections
      connect             Connect to milvus.
      exit                Exit the CLI
      filter-search       Search collection with limit 100 and distance 0.75...
      help                Show help messages
      iterator-search     Iterator search collection with limit 100 and...
      prepare-collection  Prepare a random name collection: create -> insert...
      range-search        Search collection with limit 100 and distance 0.75...
      search              Search collection with limit 100
      show-data           Show parts data of laion1b-nolang 768dim
    
    milvus_demo > connect --host=10.102.7.29
    """
