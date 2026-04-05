#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_set>
#include <random>

int f(int x) {
    return 3*x*x - 4*x + 5;
}

struct Value {
    float data;
    float grad = 0;
    std::vector<std::shared_ptr<Value>> children;
    char op = '\0';
    std::function<void()> backward = [](){};

    Value(float data) : data(data) {}
};


using Val = std::shared_ptr<Value>;

Val make_val(float data) { return std::make_shared<Value>(data); }

Val operator+(Val a, Val b) {
    auto v = std::make_shared<Value>(a->data + b->data);
    v->children = {a, b};
    v->op = '+';
    v->backward = [v, a, b]() {
        a->grad += v->grad;
        b->grad += v->grad;
    };
    return v;
}

Val operator*(Val a, Val b) {
    auto v = std::make_shared<Value>(a->data * b->data);
    v->children = {a, b};
    v->op = '*';
    v->backward = [v, a, b]() {
        a->grad += b->data * v->grad;
        b->grad += a->data * v->grad;
    };
    return v;
}

Val& operator+=(Val& a, Val b) {
    a = a + b;
    return a;
}

Val tanh_val(Val a) {
    float x = a->data;
    float t = (std::exp(2*x)-1)/(std::exp(2*x)+1);
    auto v = std::make_shared<Value>(t);
    v->children = {a};
    v->op = 't';
    v->backward = [v, a, t]() {
        a->grad += (1 - t*t) * v->grad;
    };
    return v;
}

std::ostream& operator<<(std::ostream& os, const Val& v){
    os << "Value(" << v->data << "|" << v->grad << ")";
    return os;
}

void visualize(const Val& root, std::string prefix = "", bool isLast = true){
    bool isRoot = prefix.empty();
    if (!isRoot) std::cout << prefix << (isLast ? "└── " : "├── ");
    std::cout << root;
    if (root->op) std::cout << " [" << root->op << "]";
    std::cout << std::endl;

    std::string childPrefix = prefix + (isLast ? "    " : "│   ");
    for (size_t i = 0; i < root->children.size(); ++i) {
        visualize(root->children[i], childPrefix, i == root->children.size() - 1);
    }
}

void build_topo(const Val& v, std::vector<Val>& topo, std::unordered_set<Value*>& visited) {
    if (visited.count(v.get())) return;
    visited.insert(v.get());
    for (auto& child : v->children)
        build_topo(child, topo, visited);
    topo.push_back(v);
}

void backward(const Val& root) {
    std::vector<Val> topo;
    std::unordered_set<Value*> visited;
    build_topo(root, topo, visited);
    root->grad = 1;
    for (int i = topo.size() - 1; i >= 0; --i)
        topo[i]->backward();
}

struct Neuron {
    std::vector<Val> weights;
    Val bias;

    Neuron(int nin) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        this->weights = {};
        for (size_t i = 0; i < nin; i++)
        {
            this->weights.push_back(make_val(dist(gen)));
        }
        this->bias = make_val(dist(gen));    
    }

    Val operator()(std::vector<Val> x) {
        auto result = make_val(0.0f);
        for (size_t i = 0; i < this->weights.size(); i++)
            result = result + this->weights.at(i) * x.at(i);
        return tanh_val(result + this->bias);
    }

    std::vector<Val> parameters() {
        std::vector<Val> params = {};
        params.insert(params.end(), this->weights.begin(), this->weights.end());
        params.push_back(this->bias);
        return params;
    }

};

struct Layer {
    std::vector<std::shared_ptr<Neuron>> neurons;

    Layer(int in, int out){
        this->neurons = {};
        for (size_t i = 0; i < out; i++)
        {
            this->neurons.push_back(std::make_shared<Neuron>(Neuron(in)));
        }
        
    }

    std::vector<Val> operator()(std::vector<Val> x){
        std::vector<Val> outs = {};
        for (size_t i = 0; i < this->neurons.size(); i++)
            outs.push_back((*this->neurons.at(i))(x));
        return outs;
    }

    std::vector<Val> parameters() {
        std::vector<Val> params = {};
        for (size_t i = 0; i < this->neurons.size(); i++)
        {
            auto list = this->neurons.at(i)->parameters();
            params.insert(params.end(), list.begin(), list.end());
        }
        return params;
    }
};

struct MLP {
    std::vector<std::shared_ptr<Layer>> layers;

    MLP(int in, std::vector<int> outs){
        this->layers = {};
        outs.insert(outs.begin(), in);
        for (size_t i = 0; i < outs.size()-1; i++)
        {
            this->layers.push_back(std::make_shared<Layer>(Layer(outs.at(i), outs.at(i+1))));
        }
    };
    std::vector<Val> operator()(std::vector<Val> x){
        for (size_t i = 0; i < this->layers.size(); i++)
            x = (*this->layers.at(i))(x);
        return x;
    }
    
    std::vector<Val> parameters() {
        std::vector<Val> params = {};
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            auto list = this->layers.at(i)->parameters();
            params.insert(params.end(), list.begin(), list.end());
        }
        return params;
    }
};

int main() {
    auto x1 = make_val(2), x2 = make_val(0);
    auto w1 = make_val(-3), w2 = make_val(1);
    auto b = make_val(6.8813);
    auto result = tanh_val(x1*w1 + x2*w2 + b);
    result->grad = 1;
    backward(result);
    visualize(result);

    std::vector<Val> x = {make_val(2.0f), make_val(3.0f), make_val(-1.0f)};
    auto n = MLP(3,{4,4,1});
    // auto result2 = n(x);
    // std::cout << result2.at(0) << std::endl;
    std::vector<std::vector<Val>> xs = {{make_val(2.0f), make_val(3.0f), make_val(-1.0f)},
                {make_val(3.0f), make_val(1.0f), make_val(0.5f)},
                {make_val(0.5f), make_val(1.0f), make_val(1.0f)},
                {make_val(1.0f), make_val(1.0f), make_val(-1.0f)}};
    std::vector<Val> ys = {make_val(1.0),make_val(-1.0),make_val(-1.0),make_val(1.0)};

    for (size_t i = 0; i < 100; i++)
    {
        // zero grads before forward pass
        auto params = n.parameters();
        for (auto& p : params) p->grad = 0.0f;

        // forward pass
        std::vector<Val> ypred = {};
        for (size_t j = 0; j < xs.size(); j++)
            ypred.push_back(n(xs.at(j)).at(0));

        // MSE loss
        Val loss = make_val(0.0f);
        for (size_t j = 0; j < ys.size(); j++) {
            Val diff = ypred.at(j) + (make_val(-1.0f) * ys.at(j));
            loss += diff * diff;
        }

        // backward + update
        backward(loss);
        for (auto& p : params)
            p->data += -0.01f * p->grad;

        std::cout << "Iteration: " << i+1 << std::endl;
        std::cout << "Total loss: " << loss << std::endl;
        for (size_t i = 0; i < ypred.size(); i++)
        {
            std::cout << "Prediction " << i+1 << ": " << ypred.at(i) << std::endl;
        };
    }
    return 0;
}
