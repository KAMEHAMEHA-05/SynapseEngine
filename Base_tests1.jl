using Images, FileIO
using Statistics
using Random
using Printf

# Define Tensor struct with proper initialization
mutable struct Tensor
    data::Array{Float64}
    grad::Array{Float64}
    
    function Tensor(data::Array{Float64})
        new(data, zeros(size(data)))
    end
end

# Helper function to initialize weights
function xavier_init(dims...)
    n = prod(dims[1:end-1])  # fan-in
    bound = sqrt(6.0 / n)
    return 2 * bound * rand(dims...) .- bound
end

# Operators
import Base: +, *, -, size, reshape

size(t::Tensor) = size(t.data)
reshape(t::Tensor, dims...) = Tensor(reshape(t.data, dims...))

+(a::Tensor, b::Tensor) = Tensor(a.data .+ b.data)
-(a::Tensor, b::Tensor) = Tensor(a.data .- b.data)
*(a::Tensor, b::Tensor) = Tensor(a.data * b.data)

# Abstract Neural Network Module
abstract type NNModule end

# Define ConvLayer
struct ConvLayer <: NNModule
    weights::Tensor
    biases::Tensor
    stride::Int
    padding::Int
    
    function ConvLayer(in_channels::Int, out_channels::Int, kernel_size::Int, stride::Int=1, padding::Int=0)
        weights = Tensor(xavier_init(kernel_size, kernel_size, in_channels, out_channels))
        biases = Tensor(zeros(out_channels))
        new(weights, biases, stride, padding)
    end
end

# Convolution operation
function (layer::ConvLayer)(x::Tensor)
    in_h, in_w, in_c = size(x.data)
    k_h, k_w, _, out_c = size(layer.weights.data)
    
    out_h = div(in_h - k_h + 2*layer.padding, layer.stride) + 1
    out_w = div(in_w - k_w + 2*layer.padding, layer.stride) + 1
    
    output = zeros(out_h, out_w, out_c)
    
    if layer.padding > 0
        padded = zeros(in_h + 2*layer.padding, in_w + 2*layer.padding, in_c)
        padded[1+layer.padding:end-layer.padding, 1+layer.padding:end-layer.padding, :] = x.data
    else
        padded = x.data
    end
    
    for c_out in 1:out_c
        for h in 1:out_h
            h_start = (h-1) * layer.stride + 1
            h_end = h_start + k_h - 1
            for w in 1:out_w
                w_start = (w-1) * layer.stride + 1
                w_end = w_start + k_w - 1
                patch = padded[h_start:h_end, w_start:w_end, :]
                output[h, w, c_out] = sum(patch .* layer.weights.data[:,:,:,c_out]) + layer.biases.data[c_out]
            end
        end
    end
    
    return Tensor(output)
end

# Define PoolLayer
struct PoolLayer <: NNModule
    size::Int
    stride::Int
end

# Pooling operation
function (layer::PoolLayer)(x::Tensor)
    in_h, in_w, channels = size(x.data)
    out_h = div(in_h, layer.size)
    out_w = div(in_w, layer.size)
    
    output = zeros(out_h, out_w, channels)
    
    for c in 1:channels
        for h in 1:out_h
            h_start = (h-1) * layer.size + 1
            h_end = h * layer.size
            for w in 1:out_w
                w_start = (w-1) * layer.size + 1
                w_end = w * layer.size
                output[h, w, c] = maximum(x.data[h_start:h_end, w_start:w_end, c])
            end
        end
    end
    
    return Tensor(output)
end

# Define DenseLayer
struct DenseLayer <: NNModule
    weight::Tensor
    bias::Tensor
end

function DenseLayer(input_size::Int, output_size::Int)
    weight = Tensor(xavier_init(input_size, output_size))
    bias = Tensor(zeros(output_size))
    return DenseLayer(weight, bias)
end

# Dense layer forward pass
function (layer::DenseLayer)(x::Tensor)
    return layer.weight * x + layer.bias
end

# Activation functions
function relu(x::Tensor)
    return Tensor(max.(x.data, 0))
end

function softmax(x::Tensor)
    exp_x = exp.(x.data .- maximum(x.data))
    return Tensor(exp_x ./ sum(exp_x))
end

# Loss functions
function cross_entropy_loss(pred::Tensor, target::Vector{Int})
    batch_size = size(pred.data, 2)
    losses = Float64[]
    
    for i in 1:batch_size
        probs = softmax(Tensor(pred.data[:,i]))
        push!(losses, -log(probs.data[target[i]]))
    end
    
    return mean(losses)
end

# Define CNN Model
struct SimpleCNN <: NNModule
    conv1::ConvLayer
    pool1::PoolLayer
    conv2::ConvLayer
    pool2::PoolLayer
    fc::DenseLayer
end

function SimpleCNN(num_classes::Int)
    conv1 = ConvLayer(3, 16, 3, 1, 1)
    pool1 = PoolLayer(2, 2)
    conv2 = ConvLayer(16, 32, 3, 1, 1)
    pool2 = PoolLayer(2, 2)
    
    feature_size = div(div(128, 2), 2)
    fc_input_size = 32 * feature_size * feature_size
    
    fc = DenseLayer(fc_input_size, num_classes)
    return SimpleCNN(conv1, pool1, conv2, pool2, fc)
end

# Forward pass
function (model::SimpleCNN)(x::Tensor)
    x = relu(model.conv1(x))
    x = model.pool1(x)
    x = relu(model.conv2(x))
    x = model.pool2(x)
    
    if length(size(x.data)) == 4
        batch_size = size(x.data, 4)
        x = reshape(x.data, :, batch_size)
    else
        x = reshape(x.data, :, 1)
    end
    
    x = model.fc(Tensor(x))
    return x
end

# Optimizer
struct SGD
    learning_rate::Float64
end

function step!(opt::SGD, model::SimpleCNN)
    # Update conv1
    model.conv1.weights.data .-= opt.learning_rate .* model.conv1.weights.grad
    model.conv1.biases.data .-= opt.learning_rate .* model.conv1.biases.grad
    
    # Update conv2
    model.conv2.weights.data .-= opt.learning_rate .* model.conv2.weights.grad
    model.conv2.biases.data .-= opt.learning_rate .* model.conv2.biases.grad
    
    # Update fc
    model.fc.weight.data .-= opt.learning_rate .* model.fc.weight.grad
    model.fc.bias.data .-= opt.learning_rate .* model.fc.bias.grad
    
    # Reset gradients
    model.conv1.weights.grad .= 0
    model.conv1.biases.grad .= 0
    model.conv2.weights.grad .= 0
    model.conv2.biases.grad .= 0
    model.fc.weight.grad .= 0
    model.fc.bias.grad .= 0
end

# Dataset handling
function load_and_preprocess_image(path::String, size::Tuple{Int,Int}=(128,128))
    img = load(path)
    img = imresize(img, size)
    if img isa Gray
        img = RGB.(img)
    end
    return Float64.(channelview(RGB.(img)))
end

function load_dataset(base_path::String)
    images = []
    labels = []
    
    for class_dir in readdir(base_path, join=true)
        if isdir(class_dir)
            class_name = basename(class_dir)
            for img_path in readdir(class_dir, join=true)
                if endswith(lowercase(img_path), ".jpg") || 
                   endswith(lowercase(img_path), ".jpeg") || 
                   endswith(lowercase(img_path), ".png")
                    try
                        img_data = load_and_preprocess_image(img_path)
                        push!(images, Tensor(img_data))
                        push!(labels, class_name)
                    catch e
                        println("Error loading image: $img_path")
                        println(e)
                    end
                end
            end
        end
    end
    
    return images, labels
end

function create_label_mapping(labels)
    unique_labels = unique(labels)
    return Dict(label => i for (i, label) in enumerate(unique_labels))
end

function create_batches(images, labels, batch_size)
    n = length(images)
    indices = shuffle(1:n)
    
    batches = []
    for i in 1:batch_size:n
        idx = indices[i:min(i+batch_size-1, n)]
        batch_images = cat([images[j].data for j in idx]..., dims=4)
        batch_labels = labels[idx]
        push!(batches, (Tensor(batch_images), batch_labels))
    end
    
    return batches
end

# Training function
function train!(model::SimpleCNN, train_data, val_data, opt::SGD; epochs=100, batch_size=32)
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    best_val_loss = Inf
    patience = 5
    patience_counter = 0
    
    # Create progress bar for epochs
    epoch_pbar = ProgressBar(1:epochs)
    set_description(epoch_pbar, "Training Progress")
    
    for epoch in epoch_pbar
        # Training phase
        model_state = :train
        total_loss = 0.0
        batches = create_batches(train_images, train_labels, batch_size)
        
        # Create progress bar for batches
        batch_pbar = ProgressBar(batches)
        set_description(batch_pbar, "Epoch $epoch")
        
        for (batch_x, batch_y) in batch_pbar
            # Forward pass
            pred = model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            
            # Backward pass (gradient computation)
            backward(pred, batch_y, model)
            
            # Update parameters
            step!(opt, model)
            
            total_loss += loss
            
            # Update batch progress bar with current loss
            set_postfix(batch_pbar, loss="%.4f" % loss)
        end
        
        avg_train_loss = total_loss / length(batches)
        
        # Validation phase
        model_state = :val
        val_loss = 0.0
        val_batches = create_batches(val_images, val_labels, batch_size)
        
        # Progress bar for validation
        val_pbar = ProgressBar(val_batches)
        set_description(val_pbar, "Validation")
        
        for (batch_x, batch_y) in val_pbar
            pred = model(batch_x)
            batch_loss = cross_entropy_loss(pred, batch_y)
            val_loss += batch_loss
            set_postfix(val_pbar, loss="%.4f" % batch_loss)
        end
        
        avg_val_loss = val_loss / length(val_batches)
        
        # Update epoch progress bar with losses
        set_postfix(epoch_pbar, 
                   train_loss="%.4f" % avg_train_loss,
                   val_loss="%.4f" % avg_val_loss,
                   patience="$patience_counter/$patience")
        
        # Early stopping
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0
            set_postfix(epoch_pbar, 
                       train_loss="%.4f" % avg_train_loss,
                       val_loss="%.4f" % avg_val_loss,
                       patience="$patience_counter/$patience",
                       best="yes")
        else
            patience_counter += 1
            if patience_counter >= patience
                println("\nEarly stopping triggered!")
                break
            end
        end
    end
end

# Function to format time remaining
function format_time_remaining(seconds)
    hours = div(seconds, 3600)
    minutes = div(mod(seconds, 3600), 60)
    seconds = mod(seconds, 60)
    
    if hours > 0
        return @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
    else
        return @sprintf("%02d:%02d", minutes, seconds)
    end
end

# Add this to calculate and display estimated time per epoch
function estimate_epoch_time(model::SimpleCNN, train_data, batch_size)
    train_images, train_labels = train_data
    batch = first(create_batches(train_images[1:min(length(train_images), batch_size)], 
                                train_labels[1:min(length(train_labels), batch_size)], 
                                batch_size))
    
    # Time one forward and backward pass
    start_time = time()
    pred = model(batch[1])
    loss = cross_entropy_loss(pred, batch[2])
    backward(pred, batch[2], model)
    end_time = time()
    
    # Calculate total batches per epoch
    total_batches = ceil(Int, length(train_images) / batch_size)
    
    # Estimate total time per epoch
    estimated_time = (end_time - start_time) * total_batches
    
    return estimated_time
end

# Main execution
function main()
    # Dataset paths
    base_path = "D:\\Ishaan\\Bin Arena\\CCPS skin lesions\\new balanced aug ham10000\\ISIC_2018_128X128_6705"
    train_path = joinpath(base_path, "train")
    val_path = joinpath(base_path, "validation")
    test_path = joinpath(base_path, "test")
    
    # Load datasets
    println("Loading training data...")
    train_images, train_labels = load_dataset(train_path)
    println("Loading validation data...")
    val_images, val_labels = load_dataset(val_path)
    println("Loading test data...")
    test_images, test_labels = load_dataset(test_path)
    
    # Create label mapping
    label_mapping = create_label_mapping(train_labels)
    num_classes = length(label_mapping)
    
    # Convert labels to integers
    train_labels_int = [label_mapping[label] for label in train_labels]
    val_labels_int = [label_mapping[label] for label in val_labels]
    test_labels_int = [label_mapping[label] for label in test_labels]
    
    # Initialize model and optimizer
    model = SimpleCNN(num_classes)
    optimizer = SGD(0.01)
    
    # Estimate time per epoch
    println("\nEstimating training time...")
    estimated_epoch_time = estimate_epoch_time(model, (train_images, train_labels_int), 32)
    println(@sprintf("Estimated time per epoch: %s", format_time_remaining(round(Int, estimated_epoch_time))))
    println(@sprintf("Estimated total training time: %s", 
                    format_time_remaining(round(Int, estimated_epoch_time * 100))))  # 100 epochs
    
    println("\nStarting training...")
    train!(model, (train_images, train_labels_int), 
           (val_images, val_labels_int), optimizer, 
           epochs=100, batch_size=32)
    
    println("Training completed!")
end

# Run the main function
main()