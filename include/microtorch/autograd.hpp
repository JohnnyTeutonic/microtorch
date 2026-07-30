#pragma once
// The autograd tape -- DESIGN.md phase 1a, the one genuinely new component.
//
// Reverse-mode at op granularity (settled in PHASE0_KERNEL_AUDIT.md
// section 6): a Variable owns a Matrix plus, when it was produced by an op
// under grad, the closure that scatters its gradient to its parents.
// backward() topologically sorts the tape from a scalar root and runs each
// closure once. Ownership is one-directional -- children hold shared_ptrs
// to parents, never the reverse -- so the graph is a DAG of shared_ptrs
// with no cycles to leak.
#include <functional>
#include <memory>
#include <vector>

#include "microtorch/primitives.hpp"

namespace microtorch {

class Variable;
using Var = std::shared_ptr<Variable>;

class Variable {
public:
    Matrix data;
    Matrix grad;                    // sized+zeroed on first accumulate()
    bool requires_grad = false;

    // Tape node; empty for leaves. backward_fn reads this->grad and
    // accumulates into parents' grads. It captures `this` raw -- safe
    // because the closure is a member of this Variable -- and the parents
    // as shared_ptrs, which is what keeps the upstream graph alive.
    std::vector<Var> parents;
    std::function<void()> backward_fn;

    explicit Variable(Matrix d, bool rg = false)
        : data(std::move(d)), requires_grad(rg) {}

    bool is_leaf() const { return parents.empty(); }
    void accumulate(const Matrix& g);   // grad += g (sizing on first use)
};

inline Var make_var(Matrix data, bool requires_grad = false) {
    return std::make_shared<Variable>(std::move(data), requires_grad);
}

// Reverse pass from a scalar root ([1,1] -- asserted, because seeding a
// non-scalar with ones silently computes a sum-vector-Jacobian the caller
// probably did not mean).
void backward(const Var& root);

void zero_grad(const std::vector<Var>& vars);

// no_grad scope. Ops record no tape nodes while one of these is alive.
bool grad_enabled();
class NoGrad {
public:
    NoGrad();
    ~NoGrad();
    NoGrad(const NoGrad&) = delete;
    NoGrad& operator=(const NoGrad&) = delete;

private:
    bool prev_;
};

}  // namespace microtorch
