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
                            "--data_graph_path", "{}/{}.graph".format(full_data_dir, dataset), 
                            "--true_card_path", "{}/query_graph_labels.csv".format(full_data_dir), 
                            "--train_folder", "{}/train/".format(full_data_dir),
                            "--test_folder", "{}/test/".format(full_data_dir),
                            "--query_vertex_num", "all",
                            "--saved_name", saved_result_name,
                            "--stdout_file", output_file)

if __name__ == '__main__':
    dataset = 'yeast'
    full_data_dir = '/home/lxhq/Documents/workspace_1/dataset/ml_data/{}'.format(dataset)
    output_dir = 'outputs/{}/'.format(dataset)
    epochs = [30, 60, 90, 120, 150]
    lrs = [5e-3, 3e-3, 1e-3, 8e-4, 5e-4, 3e-4, 1e-4]

    processes = []
    for epoch in epochs:
        for lr in lrs:
            saved_result_name = '{}_lr_{}_better_performance_mse_log2_dropout_scheduler'.format(epoch, lr)
            output_file = output_dir + saved_result_name + '.txt'
            command = get_command(dataset, epoch, saved_result_name, output_file, full_data_dir, lr)
            processes.append(execute_binary(command))
        
        for process in processes:
            process.wait()