using DataFrames
using CSV
using PyPlot
using LinearAlgebra
using Optim

include("simpleneural.jl")


df = CSV.read("data/data.csv", DataFrame; header = true, decimal = ',')

columns = [2,3,4,5,6,7,1]
data = Matrix{Float64}(df[!, columns])


normal = [norm(col) for col in eachcol(data)]

data = (data' ./ normal)'# ' -транспонирование; ./ - "поэлементное" деление

N_neurons = 30;
N_samples = size(data,1)

# добавляем в первый столбец вектор смещения
XX = hcat(ones(N_samples),data[:, 1:6])
YY = data[:, 7]

N_inputs = size(XX, 2) 
N_outputs = size(YY, 2)

N_weights = N_neurons*(N_inputs + N_outputs)

# y' = k1*f( k11*p1 + k12*p2 +...+ k1n*1) + k2*f( k21*p1 + k22*p2 +...+ k2n*1) + k3*f( k3k31*p1 + 2*p2 +...+ k3n*1) + ...

function fitness(p, xx::Matrix{<: Real}, yy::Vector{<: Real})
    
    func = x -> NeuralNetwork(p, x, logistic); # lambda x : NeuralNetwork

    prognose = [func(row) for row in eachrow(xx)]

    return norm(prognose - yy);
end

trainmask = rand(N_samples) .< 0.7; testmask = .!trainmask

X_train = XX[trainmask,:]
Y_train = YY[trainmask]

X_test = XX[testmask, :]
Y_test = YY[testmask,end]



f(p) = fitness(p, X_train, Y_train);
p0 = rand(N_weights)
res = optimize(f, p0, method = LBFGS(), autodiff = :forward, f_tol = 1e-5, iterations = 20000); # scipy.optimize

p1 = Optim.minimizer(res);

neuralprice = x -> NeuralNetwork(p1, x, logistic);

y_pr_train = [neuralprice(row) for row in eachrow(X_train)]
plot(Y_train); plot(y_pr_train)

y_pr_test = [neuralprice(row) for row in eachrow(X_test)]
figure();plot(Y_test) ;plot(y_pr_test)

y_pr = [neuralprice(row) for row in eachrow(XX)]
figure();plot(YY*normal[7]); plot(y_pr*normal[7])