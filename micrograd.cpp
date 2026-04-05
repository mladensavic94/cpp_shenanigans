#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_set>
#include <random>


enum class Op {None, Add, Sub, Mul, Tanh};
struct Value {
    double data;
    double grad = 0;
    std::vector<std::shared_ptr<Value>> children;
    Op op = Op::None;
    std::function<void()> backward = [](){};

    Value(double data) : data(data) {}
};


using Val = std::shared_ptr<Value>;

Val make_val(double data) { return std::make_shared<Value>(data); }

Val operator+(const Val& a, const Val& b) {
    auto v = std::make_shared<Value>(a->data + b->data);
    v->children = {a, b};
    v->op = Op::Add;
    v->backward = [vp = v.get(), a, b]() {
        a->grad += vp->grad;
        b->grad += vp->grad;
    };
    return v;
}

Val operator*(const Val& a, const Val& b) {
    auto v = std::make_shared<Value>(a->data * b->data);
    v->children = {a, b};
    v->op = Op::Mul;
    v->backward = [vp = v.get(), a, b]() {
        a->grad += b->data * vp->grad;
        b->grad += a->data * vp->grad;
    };
    return v;
}

Val operator-(const Val& a, const Val& b) {
    auto negb = (make_val(-1.0) * b);
    auto v = std::make_shared<Value>(a->data + negb->data);
    v->children = {a, negb};
    v->op = Op::Sub;
    v->backward = [vp = v.get(), a, negb]() {
        a->grad += vp->grad;
        negb->grad += vp->grad;
    };
    return v;
}

Val& operator+=(Val& a, const Val& b) {
    a = a + b;
    return a;
}

Val tanh_val(const Val& a) {
    double x = a->data;
    double e = std::exp(2*x);
    double t = (e-1)/(e+1);
    auto v = std::make_shared<Value>(t);
    v->children = {a};
    v->op = Op::Tanh;
    v->backward = [vp = v.get(), a, t]() {
        a->grad += (1 - t*t) * vp->grad;
    };
    return v;
}

std::ostream& operator<<(std::ostream& os, const Val& v){
    os << "Value(" << v->data << "|" << v->grad << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, Op op) {
    switch (op) {
        case Op::Add:  return os << "+";
        case Op::Sub:  return os << "-";
        case Op::Mul:  return os << "*";
        case Op::Tanh: return os << "t";
        default:       return os;
    }
}

void visualize(const Val& root, std::string prefix = "", bool isLast = true){
    bool isRoot = prefix.empty();
    if (!isRoot) std::cout << prefix << (isLast ? "└── " : "├── ");
    std::cout << root;
    if (root->op != Op::None) std::cout << " [" << root->op << "]";
    std::cout << std::endl;

    std::string childPrefix = prefix + (isLast ? "    " : "│   ");
    for (size_t i = 0; i < root->children.size(); ++i) {
        visualize(root->children[i], childPrefix, i == root->children.size() - 1);
    }
}

void build_topo(const Val& v, std::vector<Val>& topo, std::unordered_set<Value*>& visited) {
    if (visited.contains(v.get())) return;
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

static std::mt19937 rng{std::random_device{}()};
static std::uniform_real_distribution<double> dist(-1.0, 1.0);

struct Neuron {
    std::vector<Val> weights;
    Val bias;

    Neuron(int nin) {
        for (size_t i = 0; i < nin; i++)
        this->weights.push_back(make_val(dist(rng)));
        this->bias = make_val(dist(rng));
    }

    Val operator()(const std::vector<Val>& x) const{
        auto result = this->bias;
        for (size_t i = 0; i < this->weights.size(); i++)
            result = result + this->weights.at(i) * x.at(i);
        return tanh_val(result);
    }

    std::vector<Val> parameters() const {
        std::vector<Val> params = {};
        params.insert(params.end(), this->weights.begin(), this->weights.end());
        params.push_back(this->bias);
        return params;
    }

};

struct Layer {
    std::vector<Neuron> neurons;

    Layer(int in, int out){
        for (size_t i = 0; i < out; i++)
        {
            this->neurons.emplace_back(in);
        }
        
    }

    std::vector<Val> operator()(const std::vector<Val>& x) const{
        std::vector<Val> outs = {};
        for(auto& neuron : neurons){
            outs.push_back(neuron(x));
        }
        return outs;
    }

    std::vector<Val> parameters() const {
        std::vector<Val> params = {};
        for (size_t i = 0; i < this->neurons.size(); i++)
        {
            auto list = this->neurons.at(i).parameters();
            params.insert(params.end(), list.begin(), list.end());
        }
        return params;
    }
};

struct MLP {
    std::vector<Layer> layers;

    MLP(int in, std::vector<int> outs){
        outs.insert(outs.begin(), in);
        for (size_t i = 0; i < outs.size()-1; i++)
        {
            this->layers.push_back(Layer(outs.at(i), outs.at(i+1)));
        }
    };
    std::vector<Val> operator()(std::vector<Val> x){
        for (auto& layer : layers)
            x = layer(x);
        return x;
    }
    
    std::vector<Val> parameters() const {
        std::vector<Val> params = {};
        for (auto& layer : layers)
        {
            auto list = layer.parameters();
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
    backward(result);
    visualize(result);

    auto n = MLP(5,{6,6,6,6,6,1});
    std::vector<std::vector<Val>> xs = {
        {make_val(0.0), make_val(0.0), make_val(0.0), make_val(0.0), make_val(0.0)},
        {make_val(0.0), make_val(0.0), make_val(0.0), make_val(0.0), make_val(1.0)},
        {make_val(0.0), make_val(0.0), make_val(0.0), make_val(1.0), make_val(0.0)},
        {make_val(0.0), make_val(0.0), make_val(0.0), make_val(1.0), make_val(1.0)},
        {make_val(0.0), make_val(0.0), make_val(1.0), make_val(0.0), make_val(0.0)},
        {make_val(0.0), make_val(0.0), make_val(1.0), make_val(0.0), make_val(1.0)},
        {make_val(0.0), make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0)},
        {make_val(0.0), make_val(0.0), make_val(1.0), make_val(1.0), make_val(1.0)},
        {make_val(0.0), make_val(1.0), make_val(0.0), make_val(0.0), make_val(0.0)},
        {make_val(0.0), make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0)},
        {make_val(0.0), make_val(1.0), make_val(0.0), make_val(1.0), make_val(0.0)},
        {make_val(0.0), make_val(1.0), make_val(0.0), make_val(1.0), make_val(1.0)},
        {make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0), make_val(0.0)},
        {make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0), make_val(1.0)},
        {make_val(0.0), make_val(1.0), make_val(1.0), make_val(1.0), make_val(0.0)},
        {make_val(0.0), make_val(1.0), make_val(1.0), make_val(1.0), make_val(1.0)},
        {make_val(1.0), make_val(0.0), make_val(0.0), make_val(0.0), make_val(0.0)},
        {make_val(1.0), make_val(0.0), make_val(0.0), make_val(0.0), make_val(1.0)},
        {make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0), make_val(0.0)},
        {make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0), make_val(1.0)},
        {make_val(1.0), make_val(0.0), make_val(1.0), make_val(0.0), make_val(0.0)},
        {make_val(1.0), make_val(0.0), make_val(1.0), make_val(0.0), make_val(1.0)},
        {make_val(1.0), make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0)},
        {make_val(1.0), make_val(0.0), make_val(1.0), make_val(1.0), make_val(1.0)},
        {make_val(1.0), make_val(1.0), make_val(0.0), make_val(0.0), make_val(0.0)},
        {make_val(1.0), make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0)},
        {make_val(1.0), make_val(1.0), make_val(0.0), make_val(1.0), make_val(0.0)},
        {make_val(1.0), make_val(1.0), make_val(0.0), make_val(1.0), make_val(1.0)},
        {make_val(1.0), make_val(1.0), make_val(1.0), make_val(0.0), make_val(0.0)},
        {make_val(1.0), make_val(1.0), make_val(1.0), make_val(0.0), make_val(1.0)},
        {make_val(1.0), make_val(1.0), make_val(1.0), make_val(1.0), make_val(0.0)},
        {make_val(1.0), make_val(1.0), make_val(1.0), make_val(1.0), make_val(1.0)}};
    std::vector<Val> ys = {
        make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0),
        make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0),
        make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0),
        make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0),
        make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0),
        make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0),
        make_val(0.0), make_val(1.0), make_val(1.0), make_val(0.0),
        make_val(1.0), make_val(0.0), make_val(0.0), make_val(1.0)};

    for (size_t i = 0; i < 1000; i++)
    {
        // zero grads before forward pass
        auto params = n.parameters();
        for (auto& p : params) p->grad = 0.0;

        // forward pass
        std::vector<Val> ypred = {};
        for (size_t j = 0; j < xs.size(); j++)
            ypred.push_back(n(xs.at(j)).at(0));

        // MSE loss
        Val loss = make_val(0.0);
        for (size_t j = 0; j < ys.size(); j++) {
            Val diff = ypred.at(j) - ys.at(j);
            loss += diff * diff;
        }

        // backward + update
        backward(loss);
        for (auto& p : params)
            p->data += -0.005 * p->grad;

        std::cout << "Iteration: " << i+1 << std::endl;
        std::cout << "Total loss: " << loss << std::endl;
    }

    std::cout << "\n=== Final predictions ===" << std::endl;
    std::vector<Val> ypred_final;
    auto xpred = {make_val(1),make_val(1),make_val(1),make_val(0),make_val(0)};
        ypred_final.push_back(n(xpred).at(0));
    for (size_t j = 0; j < ypred_final.size(); j++)
        std::cout << "Sample " << j+1 << ": " << ypred_final.at(j) << std::endl;

    return 0;
}
