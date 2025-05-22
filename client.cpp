#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <torch/torch.h>
#include <chrono>
#include <cstring>
#include <unistd.h>
using namespace std;

// Define MPI message tags
#define TAG_MODEL 1
#define TAG_INDICES 2
#define TAG_GRADIENTS 3
#define TAG_COMMAND 4
#define TAG_STATUS 5
#define TAG_HEARTBEAT 6
#define TAG_STOP 8

bool running = true;
bool stop_heartbeat = false;
bool simulating_failure = false;

// Define the network architecture
struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({-1, 784});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

std::string saveToTempFile(const std::string& content, const std::string& prefix) {
    std::string filename = prefix + "_" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()) + ".tmp";
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open temp file for writing: " << filename << std::endl;
        return "";
    }
    
    file << content;
    file.close();
    return filename;
}

std::string serializeTensor(const torch::Tensor& tensor) {
    std::ostringstream oss;
    torch::save(tensor, oss);
    return oss.str();
}

torch::Tensor deserializeTensor(const std::string& serialized) {
    std::istringstream iss(serialized);
    torch::Tensor tensor;
    torch::load(tensor, iss);
    return tensor;
}

Net instantiateModel(const std::string& model_definition) {
    std::string model_file = saveToTempFile(model_definition, "model_def");
    
    if (model_file.empty()) {
        throw std::runtime_error("Failed to save model definition to temporary file");
    }
    
    Net model;
    std::remove(model_file.c_str());
    return model;
}

void loadWeights(Net& model, const std::string& weights) {
    std::istringstream iss(weights);
    std::vector<torch::Tensor> params;
    torch::load(params, iss);
    auto current_params = model.parameters();
    for (size_t i = 0; i < current_params.size(); ++i) {
        current_params[i].data().copy_(params[i].detach());
    }
}

// Updated collectGradients function
std::string collectGradients(Net& model, int num_batches){
    std::vector<torch::Tensor> gradients;
    for (const auto& param : model.parameters()) {
        if (param.grad().defined()) {
            gradients.push_back(param.grad().detach().clone() / num_batches);
        } else {
            gradients.push_back(torch::zeros_like(param));
        }
    }
    std::ostringstream oss;
    torch::save(gradients, oss);
    return oss.str();
}

int trainModel(Net& model, float learning_rate, float momentum, int batch_size, int world_rank, int world_size){
    std::cout << "[Client " << world_rank << "] Hyperparameters: lr=" << learning_rate << ", momentum=" << momentum 
              << ", batch_size=" << batch_size << std::endl;
    
    auto full_dataset = torch::data::datasets::MNIST("./mnist_dataset")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    
    auto sampler = torch::data::samplers::DistributedRandomSampler(
        full_dataset.size().value(),
        world_size,
        world_rank,
        /* shuffle */ true
    );

    auto train_loader = torch::data::make_data_loader(
        std::move(full_dataset),
        sampler,
        torch::data::DataLoaderOptions()
            .batch_size(batch_size)
    );
    
    int num_gpus = torch::cuda::device_count();
    int device_id = torch::cuda::is_available() ? (world_rank % num_gpus) : -1;
    torch::Device device(torch::cuda::is_available() ? torch::Device(torch::kCUDA, device_id) : torch::kCPU);
    model.to(device);
    
    torch::optim::SGD optimizer(
        model.parameters(), torch::optim::SGDOptions(learning_rate).momentum(momentum));
    
    model.train();
    size_t batch_idx = 0;
    float total_loss = 0;

    optimizer.zero_grad(); // Initialize gradients once


    for (auto& batch : *train_loader) {        
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        
        auto output = model.forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, targets);
        loss.backward();  // Accumulate gradients
        
        total_loss += loss.item<float>();
        
        if (++batch_idx % 10 == 0) {
            std::printf("[Client %d] Batch %ld Loss: %.4f\n", world_rank, batch_idx, loss.item<float>());
        }
    }
    float avg_loss = total_loss / batch_idx;
    std::printf("[Client %d] Training complete. Average loss: %.4f\n", world_rank, avg_loss);

    return batch_idx;
}

void heartbeat_function(int rank){
    while(running){
        if(!stop_heartbeat){
            int message = rank;
            MPI_Request request;
            MPI_Isend(&message, 1, MPI_INT, 0, TAG_HEARTBEAT, MPI_COMM_WORLD, &request);
            std::cout << "[Worker " << rank << "] Heartbeat sent" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else{
            // std::cout << "[Worker " << rank << "] Heartbeat stopped" << std::endl;
            ;
        }
    }

    // std::cout << "[Worker " << rank << "] Heartbeat stopped" << std::endl;
}

// bool check_for_exit(){
//     MPI_Status status;
//     int flag = -1;
//     MPI_Request request;
//     // std::this_thread::sleep_for(std::chrono::seconds(0.1));
//     // MPI_Irecv(&flag, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, &request);
//     // sleep(0.1);
//     MPI_Iprobe(MPI_ANY_SOURCE, TAG_STOP, MPI_COMM_WORLD, &flag, &status);
//     cout << "flag: " << flag << endl;
//     // if(flag){
//     //     int source = status.MPI_SOURCE;
//     //     int tag = status.MPI_TAG;
//     //     if(tag == TAG_STOP){
//     //         int message;
//     //         MPI_Recv(&message, 1, MPI_INT, source, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     //         return true;
//     //     }
//     // }
//     return false;
// }

int main(int argc, char* argv[]) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if (world_rank == 0) {
        std::cerr << "[Client " << world_rank << "] This process should be a client, not the server" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    std::cout << "[Client " << world_rank << "] started" << std::endl;
    torch::manual_seed(world_rank);
    std::thread heartbeat_thread(heartbeat_function, world_rank);
    int num_epochs;
    MPI_Recv(&num_epochs, 1, MPI_INT, 0, TAG_COMMAND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cout << "[Client " << world_rank << "] num_epochs: " << num_epochs << endl;
    int count = 0;
    while(true){
    if(count == num_epochs){
        std::cout << "[Client " << world_rank << "] completed training" << std::endl;
        running = false;
        heartbeat_thread.join();
        break;
    }
    count++;
    try {
        // bool exit_or_not = check_for_exit();
        // cout << "exit_or_not: " << exit_or_not << endl;
        // if(exit_or_not){
        //     std::cout << "[Client " << world_rank << "] received exit signal" << std::endl;
        //     running = false;
        //     heartbeat_thread.join();
        //     break;
        // }
        int model_def_size;
        MPI_Recv(&model_def_size, 1, MPI_INT, 0, TAG_MODEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<char> model_def_buffer(model_def_size);
        MPI_Recv(model_def_buffer.data(), model_def_size, MPI_CHAR, 0, TAG_MODEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::string model_definition(model_def_buffer.begin(), model_def_buffer.end());
        std::cout << "[Client " << world_rank << "] Received model definition (" << model_def_size << " bytes)" << std::endl;
        
        int weights_size;
        MPI_Recv(&weights_size, 1, MPI_INT, 0, TAG_MODEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<char> weights_buffer(weights_size);
        MPI_Recv(weights_buffer.data(), weights_size, MPI_CHAR, 0, TAG_MODEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::string model_weights(weights_buffer.begin(), weights_buffer.end());
        std::cout << "[Client " << world_rank << "] Received model weights (" << weights_size << " bytes)" << std::endl;
        
        Net model = instantiateModel(model_definition);
        loadWeights(model, model_weights);
        
        float hyperparams[3];
        MPI_Recv(hyperparams, 3, MPI_FLOAT, 0, TAG_COMMAND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        float learning_rate = hyperparams[0];
        float momentum = hyperparams[1];
        int batch_size = static_cast<int>(hyperparams[2]);
        
        int num_batches = trainModel(model, learning_rate, momentum, batch_size, world_rank, world_size);
        
        std::string gradients = collectGradients(model, num_batches);
        
        int status = 1;
        MPI_Send(&status, 1, MPI_INT, 0, TAG_STATUS, MPI_COMM_WORLD);
        
        int gradients_size = gradients.size();
        MPI_Send(&gradients_size, 1, MPI_INT, 0, TAG_GRADIENTS, MPI_COMM_WORLD);
        MPI_Send(gradients.c_str(), gradients_size, MPI_CHAR, 0, TAG_GRADIENTS, MPI_COMM_WORLD);
        
        std::cout << "[Client " << world_rank << "] completed training and sent gradients" << std::endl;
        if(simulating_failure && world_rank == 1){
            stop_heartbeat = true;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[Client " << world_rank << "] error: " << e.what() << std::endl;
        
        int status = 0;
        MPI_Send(&status, 1, MPI_INT, 0, TAG_STATUS, MPI_COMM_WORLD);
    }
    }    

    // MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Finalize();
    return 0;
}