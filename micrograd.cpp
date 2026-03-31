#include <iostream>
#include <vector>

int f(int x) {
    return 3*x*x - 4*x + 5;
}

class Value {
public:
    float data;
    std::vector<Value> children;
    char op;

    Value(float data) : data(data), children({}), op('\0') {}
    Value(float data, std::vector<Value> children, char op) : data(data), children(children), op(op) {}

    bool operator==(const Value& other) const {
        return data == other.data;
    }

    Value operator+(const Value& other) const {
        return Value(data+other.data, std::vector<Value> {*this, other}, '+');
    }
    Value operator*(const Value& other) const {
        return Value(data*other.data, std::vector<Value> {*this, other}, '*');
    }
};

std::ostream& operator<<(std::ostream& os, const Value& v){
    os << "Value(" << v.data << ")";
    return os;
}

std::vector<float> arrange(float s, float e, float step) {
    std::vector<float> v = {};
    for (float i = s; i < e; i += step)
    {
        v.push_back(i);
    }
    
    return v;
}

void visualize(const Value& root, std::string prefix = "", bool isLast = true){
    bool isRoot = prefix.empty();
    if (!isRoot) std::cout << prefix << (isLast ? "└── " : "├── ");
    std::cout << "Value(" << root.data << ")";
    if (root.op) std::cout << " [" << root.op << "]";
    std::cout << std::endl;

    std::string childPrefix = prefix + (isLast ? "    " : "│   ");
    for (size_t i = 0; i < root.children.size(); ++i) {
        visualize(root.children[i], childPrefix, i == root.children.size() - 1);
    }
}

int main() {
    int x = 3;
    std::cout << "scale of " << x << " is " << f(x) << std::endl;

    std::vector<float> v = arrange(-5, 5, 0.25);
    std::vector<Value> vals = {};
    for (float val : v) {
        vals.push_back(Value(val));
    }
    std::cout << v[1] << " " << v[3] << std::endl;
    std::cout << v[1]*v[3] << std::endl;

    Value a(2), b(3), c(4);
    Value result = (a + b) * c + Value(1);
    visualize(result);
    return 0;
}