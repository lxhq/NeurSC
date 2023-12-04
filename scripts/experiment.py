from subprocess import Popen, PIPE

def execute_binary(args):
    process = Popen(' '.join(args), shell=True, stdout=PIPE, stderr=PIPE)
    # (std_output, std_error) = process.communicate()
    return process
    # process.wait()
    # rc = process.returncode
    # if std_error:
    #     print(std_error.decode(), flush=True)
    #     exit(-1)
    # return rc, std_output, std_error

def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments

def get_command(dataset, epochs, saved_result_name, output_file, full_data_dir, lr):
    return generate_args("python3 src/main.py",
                            "--learning_rate", str(lr),
                            "--num_epoch", str(epochs), 
                            "--sample_method", "induced", 
                            "--share_net", '\"\"',
                            "--model_name", "wasserstein", 
                            "--device", "cuda",
                            "--graph_file", dataset,
                            "--data_graph_path", "{}/data_graph/{}.graph".format(full_data_dir, dataset), 
                            "--true_card_path", "{}/query_graph_cards.csv".format(full_data_dir), 
                            "--train_names", "{}/train.csv".format(full_data_dir),
                            "--test_names", "{}/test.csv".format(full_data_dir),
                            "--query_graph_dir", "{}/query_graph/".format(full_data_dir),
                            "--query_vertex_num", "all",
                            "--saved_name", saved_result_name,
                            "--stdout_file", output_file)

if __name__ == '__main__':
    datasets = ['youtube', 'yeast']
    epochs = [150]
    lrs = [1e-3]
    repeat = 10
    for dataset in datasets:
        full_data_dir = '/home/lxhq/Documents/workspace/dataset/{}'.format(dataset)
        output_dir = 'outputs/{}/'.format(dataset)
        for epoch in epochs:
            for lr in lrs:
                processes = []
                for idx in range(repeat):
                    saved_result_name = 'mse_log2_dropout_scheduler_{}_{}_{}_{}'.format(dataset, epoch, lr, idx)
                    output_file = output_dir + saved_result_name + '.txt'
                    command = get_command(dataset, epoch, saved_result_name, output_file, full_data_dir, lr)
                    processes.append(execute_binary(command))
                    if (idx + 1) % 5 == 0:
                        for process in processes:
                            process.wait()
                        processes = []