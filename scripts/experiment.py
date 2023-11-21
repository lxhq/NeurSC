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

def get_command(dataset, epochs, saved_results_name, full_data_dir):
    return generate_args("python3 src/main.py",
                            "--num_epoch", str(epochs), 
                            "--model_name", "wasserstein", 
                            "--graph_file", dataset,
                            "--data_graph_path", "{}/{}.graph".format(full_data_dir, dataset), 
                            "--true_card_path", "{}/query_graph_labels.csv".format(full_data_dir), 
                            "--train_folder", "{}/train/".format(full_data_dir),
                            "--test_folder", "{}/test/".format(full_data_dir),
                            "--saved_name", saved_results_name)

if __name__ == '__main__':
    dataset = 'yeast'
    epochs = [30, 50, 70, 90, 100, 120, 140, 150]
    full_data_dir = '/home/lxhq/Documents/workspace_1/dataset/ml_data/{}'.format(dataset)
    processes = []
    for epoch in epochs:
        saved_result_name = 'baseline'
        command = get_command(dataset, epoch, saved_result_name, full_data_dir)
        processes.append(execute_binary(command))
    
    for process in processes:
        process.wait()