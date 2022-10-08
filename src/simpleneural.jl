function logistic(x::T) where T <: Real
    return 1/(1+exp(-x))
end

function ReLU(x::T) where T <: Real
    if x < 0
        return 0
    else
        return x
    end
end



function NeuralNetwork(p, x, activation_func) where T <: Real
    M = size(x, 1);
    K = size(p, 1);
    @assert (K%(M+1)==0) "DIMENSIONS OF {p} AND {x} MISMATCH"
    N = K ÷ (M+1);
    
    W1 = reshape(p[1:N*M], M, N);
    W2 = reshape(p[(N*M +1): end], N,1);
    
    return sum(activation_func.((x')*W1) * W2)
end
