#include "microtorch/autograd.hpp"

#include <stdexcept>
#include <unordered_set>

namespace microtorch {

void Variable::accumulate(const Matrix& g) {
    if (grad.rows() == 0) {
        grad = Matrix(data.rows(), data.cols());  // zero-filled by ctor
    }
    if (g.rows() != grad.rows() || g.cols() != grad.cols()) {
        throw std::runtime_error("accumulate: gradient shape mismatch");
    }
    grad += g;
}

namespace {

// Post-order DFS over parents. Iterative, because a deep tape (15k-step
// training graphs are the eventual customer) must not be bounded by the C++
// call stack.
void topo(const Var& root, std::vector<Variable*>& order) {
    std::unordered_set<Variable*> seen;
    std::vector<std::pair<Variable*, size_t>> stack{{root.get(), 0}};
    seen.insert(root.get());
    while (!stack.empty()) {
        auto& [node, next] = stack.back();
        if (next < node->parents.size()) {
            Variable* p = node->parents[next++].get();
            if (seen.insert(p).second) stack.emplace_back(p, 0);
        } else {
            order.push_back(node);
            stack.pop_back();
        }
    }
}

thread_local bool g_grad_enabled = true;

}  // namespace

void backward(const Var& root) {
    if (root->data.rows() != 1 || root->data.cols() != 1) {
        throw std::runtime_error("backward: root must be a [1,1] scalar (compose with mean/sum)");
    }
    Matrix seed(1, 1);
    seed(0, 0) = 1.0f;
    root->accumulate(seed);

    std::vector<Variable*> order;  // post-order: leaves first
    topo(root, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {  // root first
        if ((*it)->backward_fn && (*it)->grad.rows() != 0) {
            (*it)->backward_fn();
        }
    }
}

void zero_grad(const std::vector<Var>& vars) {
    for (const auto& v : vars) {
        if (v->grad.rows() != 0) v->grad.fill(0.0f);
    }
}

bool grad_enabled() {
    return g_grad_enabled;
}

NoGrad::NoGrad() : prev_(g_grad_enabled) {
    g_grad_enabled = false;
}
NoGrad::~NoGrad() {
    g_grad_enabled = prev_;
}

}  // namespace microtorch
