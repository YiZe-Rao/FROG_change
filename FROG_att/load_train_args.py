# Due to the limitations of memory, multiple parameters need to be adjusted to ensure no OOM occurs
def load_train_args(args):
    dataset, task = args.dataset, args.task
    print(f'loading... | {dataset} | {task}')
    
    if dataset == 'rel-amazon':
        raise NotImplementedError("rel-amazon dataset is not supported.")
    
    elif dataset == 'rel-avito':
        if task == 'user-visits': 
            args.batch_size = 1024
            args.dropout = 0.3
            args.lr = 0.001
            args.num_neighbors = 64
            args.channels = 32
        elif task == 'user-clicks':
            args.batch_size = 512
            args.dropout = 0.5
            args.lr = 0.0005
            args.num_layers = 1
            args.epochs = 20
            args.scheduler = True
        elif task == 'ad-ctr':
            args.batch_size = 128
            args.epochs = 20
            args.lr = 0.001
            args.num_neighbors = 64
            args.channels = 32
        elif task == 'user-ad-visit':
            args.batch_size = 256
            args.num_neighbors = 64
            args.num_layers = 8

    elif dataset == 'rel-event':
        args.batch_size = 128
        if task == 'user-repeat':
            args.dropout = 0.5
        elif task == 'user-ignore':
            args.lr = 0.0001
            args.dropout = 0.5
        elif task == 'user-attendance':
            args.epochs = 20
            args.MAE = 0.9999
            args.gamma = 0.

    elif dataset == 'rel-f1':
        args.batch_size = 64
        if task == 'driver-top3':
            args.num_layers = 1
            args.lr = 0.005
            args.dropout = 0.3 # 0.5
        elif task == 'driver-dnf':
            args.lr = 0.005
            args.epochs = 15
        elif task == 'driver-position':
            # The recommended batch of 256 exceeds the available 16 GiB T4 memory.
            args.batch_size = 64
            args.num_layers = 1
            args.lr = 0.0005
            args.dropout = 0.5
            args.num_neighbors = 64
            
    elif dataset == 'rel-hm':
        if task == 'user-churn':
            args.channels = 32
            args.batch_size = 512
            args.lr = 0.001
            args.epochs = 20
            args.scheduler = True
            args.MAE = 0.95
        elif task == 'item-sales':
            args.batch_size = 256
            args.lr = 0.001
            args.epochs = 20
            args.scheduler = True
        elif task == 'user-item-purchase':
            args.num_neighbors = 64
            args.batch_size = 512
            args.channels = 64
            args.num_layers = 3
            args.lr = 0.0005
            args.dropout = 0.3

    elif dataset == 'rel-stack':
        args.num_neighbors = 64
        args.channels = 32
        if task == 'user-badge':
            args.batch_size = 1024
            args.lr = 0.005  #temp
            args.epochs = 20
            args.scheduler = True
        elif task == 'user-engagement':
            args.dropout = 0.3
            args.lr = 0.001
            args.batch_size = 1024
            args.epochs = 20
            args.scheduler = True
        elif task == 'post-votes':
            args.batch_size = 512
            args.lr = 0.0005
            args.epochs = 20
            args.scheduler = True
            args.warmup = 5
        elif task == 'user-post-comment':
            args.batch_size = 128
            args.channels = 64
            args.lr = 0.005
            args.num_layers = 6
            args.scheduler = True
        elif task == 'post-post-related':
            args.batch_size = 512
            args.channels = 64
            args.lr = 0.005
            args.dropout = 0.2

    elif dataset == 'rel-trial':
        if task == 'study-outcome':
            args.aggr = 'mean'
            args.batch_size = 256
            args.num_neighbors = 64
            args.channels = 64
            args.lr = 0.001
            args.dropout = 0.2
        elif task == 'site-success':
            args.aggr = 'mean'
            args.batch_size = 128
            args.channels = 64
            args.lr = 0.001
            args.dropout = 0.3
        elif task == 'study-adverse':
            args.batch_size = 128
            args.num_neighbors = 64
            args.channels = 64
            args.lr = 0.001
            args.dropout = 0.3
            args.epochs = 20
        elif task == 'condition-sponsor-run':
            args.batch_size = 128
            args.num_neighbors = 64
            args.channels = 128
            args.num_layers = 6
            args.lr = 0.0001
        elif task == 'site-sponsor-run':
            args.batch_size = 256
            args.num_neighbors = 64
            args.num_layers = 4
            args.lr = 0.005
    return args
