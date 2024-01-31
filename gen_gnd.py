from pymilvus import connections, Collection
import numpy as np

if __name__ == '__main__':
    host = "10.102.7.29"
    coll_name = "Zdemo_ddemo_idemo_c"
    connections.connect(host=host)
    c = Collection(coll_name)
    output_fields = ["pk", "width", "height"]

    # rebuild index
    index_params = c.index().params
    c.release()
    c.drop_index()
    index_params = {"metric_type": "COSINE", "index_type": "FLAT", "params": {}}
    c.create_index("float32_vector", index_params=index_params, _async=False)
    c.load()

    # cacl limit 100 ids
    query_vec = np.load("data/query.npy")
    param_limit_100 = {
        "metric_type": "COSINE", "params": {}
    }
    search_res_1 = c.search(data=[query_vec], anns_field='float32_vector', param=param_limit_100, limit=100, output_fields=output_fields)
    print(len(search_res_1[0]))
    print(search_res_1)
    np.save('data/ids_limit_100.npy', search_res_1[0].ids)

    # cacl range search
    param_range = {
        "metric_type": "COSINE", "params": {"radius": 0.7, "range_filter": 1.0}
    }
    search_res_2 = c.search(data=[query_vec], anns_field='float32_vector', param=param_range, limit=100, output_fields=output_fields)
    print(len(search_res_2[0]))
    print(search_res_2[0].ids)
    np.save('data/ids_range_0.75_1.npy', search_res_2[0].ids)

    # cacl range and expr
    param_range_expr = {
        "metric_type": "COSINE", "params": {"radius": 0.7, "range_filter": 1.0}
    }
    search_res_3 = c.search(data=[query_vec], anns_field='float32_vector', expr='width < 500', param=param_range_expr,
                            limit=100)
    print(len(search_res_3[0]))
    print(search_res_3[0].ids)
    np.save('data/ids_range_expr.npy', search_res_3[0].ids)

    # cacl search iterator
    param_iterator = {
        "metric_type": "COSINE",
        "params": {"radius": 0.7, "range_filter": 1.0}
    }
    search_iterator = c.search_iterator(data=[query_vec], anns_field='float32_vector', expr='width < 500',
                                        param=param_iterator,
                                        batch_size=10,
                                        limit=100)
    result_ids = []
    while True:
        res = search_iterator.next()
        if len(res) == 0:
            print("search iteration finished, close")
            search_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
            result_ids.append(res[i].id)
    print(len(result_ids))
    np.save('data/ids_expr_iterator.npy', result_ids)

    # restore collection
    c.release()
    c.create_index("float32_vector", index_params=index_params, _async=False)
    c.load()
