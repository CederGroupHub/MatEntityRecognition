# Speed up with multiple computers/processors

1. Either one of the two main scripts provided can be used for acceleration: 
    `MER_multiprocess_json.py` for local storage
    and
    `MER_multiprocess_mongo.py` for cloud storage with mongo db.
    It is recommended to use mongo db for tons of data.
    
2. Both scripts splits a big task into multiple sub-tasks and monitor the status of all the sub-tasks with a local file or a mongo db collection.
    
3. Each process is coordinated with the status monitoring code to avoid duplication when multiple processes exist.
    
4. You can edit three parts accordingly to adapt to your situation.

    * The query filter to get the text data to be processed. 
    Also edit other code around for more sophisticate query.
        
            query_filter_str= json.dumps({
                'classification': {
                    '$exists': True,
                    '$nin': ['something_else', None],
                }
            })

    * The funtion `pipeline_in_one_thread()` to process the text data. 
    
    * The scheme to save the processed data.
    Also edit other code around for more sophisticate scheme.
        
            db_col.update_one(
                {'paragraph_id': meta_id},
                {
                    '$set': {
                        'MER_result': r_para,
                        'MER_version': MER.__version__,
                    }
                },
                upsert=True,
            ) 
