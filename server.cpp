#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <torch/torch.h>
#include <torch/serialize.h>

using namespace std;
// Define MPI message tags
#define TAG_MODEL 1
#define TAG_INDICES 2
#define TAG_GRADIENTS 3
#define TAG_COMMAND 4
#define TAG_STATUS 5
#define TAG_HEARTBEAT 6
#define TAG_CLIENT_DONE 7
#define TAG_STOP 8

// Heartbeat Configuration
int HEARTBEAT_TIMEOUT = 3;

std::atomic<bool> stop_heartbeat(false);
std::atomic<bool> running(true); 

int world_size;
std::unordered_map<int, double> worker_last_heartbeat;
struct Node
{
    int rank;
    int last_heartbeat_time;
    bool is_available;
};

// for metadata node
vector<Node> nodes;
vector<bool> is_available_array;

struct TrainingConfig {
    int num_epochs = 30;
    float learning_rate = 0.1;
    float momentum = 0.0;
    int batch_size = 64;
    int total_samples = 60000; // MNIST training set size
};

int compute_accuracy(const torch::Tensor& predictions, const torch::Tensor& labels) {
    auto pred_labels = predictions.argmax(1);
    return (pred_labels == labels).sum().item<int>();
}

// Define the same network architecture as clients
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

// Helper function to serialize model parameters
std::string serializeParameters(const Net& model) {
    std::vector<torch::Tensor> params;
    for (const auto& param : model.parameters()) {
        params.push_back(param.detach().clone());
    }
    std::ostringstream oss;
    torch::save(params, oss);
    return oss.str();
}

// Helper function to generate client indices
std::vector<int> generateIndices(int start, int count, int total) {
    std::vector<int> indices;
    indices.reserve(count);
    for (int i = 0; i < count; ++i) {
        indices.push_back((start + i) % total);
    }
    return indices;
}

void master_thread_for_heartbeat(){
    // for (auto node : nodes) {
        // cout << "Worker " << node.rank << " is available: " << node.is_available << endl;
    // }
    // while(1){
    //     ;
    // }

    while(true){
        if(running){
            MPI_Status status;
            int flag;
            // std::this_thread::sleep_for(std::chrono::seconds(0.1));
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_HEARTBEAT, MPI_COMM_WORLD, &flag, &status);
            if(flag){
                int source = status.MPI_SOURCE;
                int tag = status.MPI_TAG;
                if(tag == TAG_HEARTBEAT){
                    int message;
                    MPI_Recv(&message, 1, MPI_INT, source, TAG_HEARTBEAT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // std::cout << "Heartbeat received from Worker " << source << ": " << message << "\n";
                    nodes[source-1].last_heartbeat_time = MPI_Wtime();
                }
            }

            double current_time = MPI_Wtime();
            for(auto node : nodes) {
                if(current_time - node.last_heartbeat_time >= HEARTBEAT_TIMEOUT) {
                    // std::cout << "Worker " << node.rank << " failed to send heartbeat (Timeout). " << is_available_array[node.rank] << std::endl;
                    is_available_array[node.rank] = false;
                    node.is_available = false;
                }
                else{
                    // node.is_available = true;
                    is_available_array[node.rank] = true;
                    // cout << "Worker " << node.rank << " is available" << "time" << current_time - node.last_heartbeat_time << endl;
                    ;
                }
            }
        }
        else{
            // cout << "Master thread stopped" << endl;
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    // MPI_Init(&argc, &argv);
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if (world_size < 2) {
        std::cerr << "This application requires at least 2 MPI processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (world_rank == 0) {
        for(int i=1;i<world_size;i++){
            Node node;
            node.rank = i;
            node.last_heartbeat_time = 0;
            node.is_available = true;
            is_available_array.push_back(true);
            nodes.push_back(node);
        }
        is_available_array.push_back(true);
        thread heartbeat_thread(master_thread_for_heartbeat);
        std::cout << "Server starting with " << world_size - 1 << " clients" << std::endl;
        
        // Initialize server model and training configuration
        Net server_model;
        TrainingConfig config;
        for (int i = 1; i < world_size; ++i) {
            MPI_Send(&config.num_epochs, 1, MPI_INT, i, TAG_COMMAND, MPI_COMM_WORLD);
        }
        auto test_dataset = torch::data::datasets::MNIST("./mnist_dataset", 
            torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

        auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(1000).workers(2));
            
        int num_clients = world_size - 1;
        int samples_per_client = config.total_samples / num_clients;
        
        // Initial model parameters
        std::string model_weights = serializeParameters(server_model);
        // cout << model_weights << endl;
        // exit(0);
        
        // Model definition (same as client code)
        std::string model_definition = R"(
#include <torch/torch.h>

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
)";

        // Main training loop
        for (int epoch = 1; epoch <= config.num_epochs; ++epoch) {
            std::cout << "\nStarting epoch " << epoch << std::endl;
            
            // 1. Send current model to all clients
            for (int client = 1; client < world_size; ++client) {
                // Send model definition
                // cout << nodes[client-1].rank << " hi " << nodes[client-1].is_available << endl;
                // if(nodes[client-1].is_available == false){
                if (is_available_array[client] == false){
                    std::cout << "Worker " << client << " is not available" << std::endl;
                    continue;
                }
                int model_def_size = model_definition.size();
                MPI_Send(&model_def_size, 1, MPI_INT, client, TAG_MODEL, MPI_COMM_WORLD);
                MPI_Send(model_definition.c_str(), model_def_size, MPI_CHAR, client, TAG_MODEL, MPI_COMM_WORLD);
                
                // Send model weights
                int weights_size = model_weights.size();
                MPI_Send(&weights_size, 1, MPI_INT, client, TAG_MODEL, MPI_COMM_WORLD);
                MPI_Send(model_weights.c_str(), weights_size, MPI_CHAR, client, TAG_MODEL, MPI_COMM_WORLD);
            }
            std::cout << "Sent model to all clients" << std::endl;

            // 2. Distribute data indices
            // for (int client = 1; client < world_size; ++client) {
            //     int client_offset = (client - 1) * samples_per_client;
            //     std::vector<int> indices = generateIndices(client_offset, samples_per_client, config.total_samples);

            //     int size_of_indices = indices.size();
            //     MPI_Send(&size_of_indices, 1, MPI_INT, client, TAG_INDICES, MPI_COMM_WORLD);
            //     MPI_Send(indices.data(), indices.size(), MPI_INT, client, TAG_INDICES, MPI_COMM_WORLD);
            // }

            // 3. Send training commands with hyperparameters
            for (int client = 1; client < world_size; ++client) {
                // if(nodes[client-1].is_available == false){
                if (is_available_array[client] == false){
                    std::cout << "Worker " << client << " is not available" << std::endl;
                    continue;
                }
                float hyperparams[3] = {config.learning_rate, config.momentum, static_cast<float>(config.batch_size)};
                MPI_Send(hyperparams, 3, MPI_FLOAT, client, TAG_COMMAND, MPI_COMM_WORLD);
            }

            // 4. Collect gradients from clients
            std::vector<std::vector<torch::Tensor>> all_gradients;
            for (int client = 1; client < world_size; ++client) {
                // if(nodes[client-1].is_available == false){
                if (is_available_array[client] == false){
                    std::cout << "Worker " << client << " is not available" << std::endl;
                    continue;
                }
                int status;
                MPI_Recv(&status, 1, MPI_INT, client, TAG_STATUS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (status == 1) {
                    int gradients_size;
                    MPI_Recv(&gradients_size, 1, MPI_INT, client, TAG_GRADIENTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    std::vector<char> gradient_buffer(gradients_size);
                    MPI_Recv(gradient_buffer.data(), gradients_size, MPI_CHAR, client, TAG_GRADIENTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    std::istringstream iss(std::string(gradient_buffer.begin(), gradient_buffer.end()));
                    std::vector<torch::Tensor> grads;
                    torch::load(grads, iss);
                    all_gradients.push_back(grads);
                }
            }

            // 5. Aggregate gradients and update model
            if (!all_gradients.empty()) {
                // Initialize average gradients
                std::vector<torch::Tensor> avg_gradients;
                std::cout << "Aggregating gradients from " << all_gradients.size() << " clients" << std::endl;
                for (size_t i = 0; i < all_gradients[0].size(); ++i) {
                    avg_gradients.push_back(torch::zeros_like(all_gradients[0][i]).to(torch::kCPU));
                }
                std::cout << "Initialized average gradients" << std::endl;
                // Sum gradients from all clients
                for (const auto& client_grads : all_gradients) {
                    for (size_t i = 0; i < client_grads.size(); ++i) {
                        avg_gradients[i] += client_grads[i].to(torch::kCPU);
                    }
                }
                std::cout << "Summed gradients from all clients" << std::endl;
                // Average gradients
                for (auto& grad : avg_gradients) {
                    int size_of_gradient_array = all_gradients.size();
                    grad /= size_of_gradient_array;
                }
                std::cout << "Averaged gradients" << std::endl;
                // Apply gradients to server model
                auto params = server_model.parameters();
                for (size_t i = 0; i < params.size(); ++i) {
                    if (params[i].requires_grad()) {
                        params[i].mutable_grad() = avg_gradients[i].clone().to(params[i].device());
                    }
                }
                std::cout << "Applied averaged gradients to server model" << std::endl;
                // Update model parameters
                torch::optim::SGD optimizer(
                    server_model.parameters(),
                    torch::optim::SGDOptions(config.learning_rate).momentum(config.momentum)
                );
                std::cout << "Device of model parameters: " << server_model.parameters()[0].device() << " and device of gradients: " << avg_gradients[0].device() << std::endl;
                optimizer.step();
                std::cout << "Updated server model parameters" << std::endl;

                // Update model weights for next epoch
                model_weights = serializeParameters(server_model);
                std::cout << "Model updated with averaged gradients" << std::endl;
            }

            // Test the model
            server_model.eval();
            torch::NoGradGuard no_grad;
            float test_loss = 0;
            int correct = 0;
            size_t num_test_samples = 0;

            for (const auto& batch : *test_loader) {
                auto data = batch.data;
                auto targets = batch.target;
                num_test_samples += batch.data.size(0);
                
                auto output = server_model.forward(data);
                test_loss += torch::nn::functional::cross_entropy(
                    output, targets, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum))
                    .item<float>();
                
                auto pred = output.argmax(1);
                correct += pred.eq(targets).sum().item<int>();
            }

            test_loss /= num_test_samples;
            float test_accuracy = 100.0 * correct / num_test_samples;

            std::printf(
                "Test set: Average loss: %.4f Accuracy: %.2f%%\n\n",
                test_loss, test_accuracy);

            std::cout << "Completed epoch " << epoch << "/" << config.num_epochs << std::endl;
            // MPI_Barrier(MPI_COMM_WORLD);

        }

        // Final model save (example)
        // torch::save(server_model, "final_model.pt");
        std::cout << "\nTraining complete!" << std::endl;
        // for (int client = 1; client < world_size; ++client) {
        //     int stop_signal = 1;
        //     MPI_Send(&stop_signal, 1, MPI_INT, client, TAG_STOP, MPI_COMM_WORLD);
        // }
        running = false;
        heartbeat_thread.join();
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
