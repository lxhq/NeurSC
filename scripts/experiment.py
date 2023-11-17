from subprocess import Popen, PIPE

def execute_binary(args):
    process = Popen(' '.join(args), shell=True, stdout=PIPE, stderr=PIPE)
    (std_output, std_error) = process.communicate()
    process.wait()
    rc = process.returncode
    if std_error:
        print(std_error.decode(), flush=True)
        exit(-1)
    return rc, std_output, std_error

def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments

def run_train(dataset, epochs, saved_results_name, output_file, full_data_dir):
    command = generate_args("python3 src/main.py",
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
                            "--saved_name", saved_results_name,
                            "--stdout_file", output_file)
    execute_binary(command)

if __name__ == '__main__':
    dataset = 'yeast'
    epochs = 100
    full_data_dir = '/home/lxhq/Documents/workspace_1/dataset/ml_data/{}'.format(dataset)
    output_dir = 'outputs/{}/'.format(dataset)

    for idx in range(1):
        model_name = 'baseline_{}'.format(idx)
        output_file = output_dir + '{}_{}'.format(epochs, model_name) + '.txt'
        run_train(dataset, epochs, model_name, output_file, full_data_dir)