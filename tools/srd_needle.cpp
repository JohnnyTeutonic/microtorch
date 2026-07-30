// Needle-in-haystack associative recall: the sharpest test of V1's claim
// (SPARSE_ATTENTION.md). Linear attention's compressed state should fail
// precise KV recall; exact attention should succeed; SRD graduates only if
// it matches exact AND its gate concentrates on retrieval-critical
// positions.
//
//   srd_needle [steps=600] [T=256] [d=128] [csv_prefix=/tmp/srd_needle]
//              [batch=1] [npairs=8] [nkeys=64]
//
// Task (synthetic vocab of 256 symbols):
//   [filler | 8 x (key value) pairs | random filler ... | QUERY key_j]
//   next-token target at the final position is val_j.
// Fresh random sequences every step (infinite data); a FIXED 32-sequence
// probe set is evaluated every 25 steps under NoGrad: answer CE, answer
// argmax accuracy, and the SRD gate profile (gate at the query/key tail
// vs gate over filler) per lane.
//
// Pre-registered (before the first run):
//   exact answer-accuracy high (>0.8 by step 600); kimi materially lower;
//   srd close to exact; srd_f below srd; SRD tail-gate > filler-gate.
//
// Four lanes on identical sequences, checkpoint/resume identical to
// srd_parity (chunked execution; optimizer moments reset per chunk,
// identically for every lane).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "microtorch/safetensors.hpp"
#include "parity_model.hpp"

using namespace microtorch;
using parity::AttnKind;
using parity::ParityLM;

namespace {

constexpr int VOCAB = 256, QUERY = 1;
constexpr int KEY0 = 2, NKEYS = 64;      // keys  [2, 66)
constexpr int VAL0 = 66;                 // vals  [66, 130)
constexpr int FILL0 = 130, NFILL = 120;  // fill  [130, 250)

// Difficulty knobs (control-first calibration: three runs where exact
// never left the log-nkeys plateau mean the comparison needs a config
// the control can pass first). Defaults reproduce the original task.
// Vocab layout is unchanged; smaller nkeys/npairs just restrict draws.
int g_npairs = 8, g_nkeys = 64;

// One sequence of length T+1; x = seq[0..T-1], y = seq[1..T].
std::vector<int> make_seq(size_t T, std::mt19937& rng) {
    std::uniform_int_distribution<int> fill(FILL0, FILL0 + NFILL - 1);
    std::vector<int> keys(g_nkeys);
    for (int i = 0; i < g_nkeys; ++i) keys[i] = i;
    std::shuffle(keys.begin(), keys.end(), rng);   // distinct keys per seq

    std::vector<int> seq(T + 1);
    seq[0] = fill(rng);
    std::vector<int> val_of(g_npairs);
    for (int p = 0; p < g_npairs; ++p) {
        val_of[p] = VAL0 + (rng() % g_nkeys);
        seq[1 + 2 * p] = KEY0 + keys[p];
        seq[2 + 2 * p] = val_of[p];
    }
    for (size_t i = 1 + 2 * g_npairs; i + 2 < T; ++i) seq[i] = fill(rng);
    const int j = rng() % g_npairs;
    seq[T - 2] = QUERY;
    seq[T - 1] = KEY0 + keys[j];
    seq[T] = val_of[j];                     // the answer
    return seq;
}

struct Lane {
    const char* name;
    ParityLM model;
    nn::AdamW opt;
    bool is_srd;
    Lane(const char* n, AttnKind k, size_t T, size_t d, unsigned seed, float lr)
        : name(n), model(k, VOCAB, d, /*heads=*/4, T, seed),
          opt(model.parameters(), lr), is_srd(k == AttnKind::SRD) {}
};

}  // namespace

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 600;
    const size_t T = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 256;
    const size_t d = argc > 3 ? static_cast<size_t>(std::atoi(argv[3])) : 128;
    const std::string prefix = argc > 4 ? argv[4] : "/tmp/srd_needle";
    // Batch>1 is the pre-registered escalation after two runs where no lane
    // (incl. exact) escaped the log-64 plateau at 1 seq/step: B sequences
    // per optimizer step, gradients accumulated, identical batch per lane.
    const int B = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;
    if (argc > 6) g_npairs = std::max(1, std::min(8, std::atoi(argv[6])));
    if (argc > 7) g_nkeys = std::max(g_npairs, std::min(64, std::atoi(argv[7])));
    const float lr = 3e-3f, lambda_gate = 0.05f;
    const int PROBE_EVERY = 25, NPROBE = 32;

    std::vector<Lane> lanes;
    lanes.emplace_back("exact", AttnKind::EXACT, T, d, 7, lr);
    lanes.emplace_back("kimi", AttnKind::KIMI, T, d, 7, lr);
    lanes.emplace_back("srd", AttnKind::SRD, T, d, 7, lr);
    lanes.emplace_back("srd_f", AttnKind::SRD, T, d, 7, lr);
    lanes[3].model.set_falsifier(true);
    for (auto& l : lanes) l.model.train();

    // Fixed held-out probe set.
    std::mt19937 probe_rng(9999);
    std::vector<std::vector<int>> probes;
    for (int i = 0; i < NPROBE; ++i) probes.push_back(make_seq(T, probe_rng));

    const char* ck = std::getenv("SRD_CKPT_DIR");
    const std::string ckpt_dir = ck ? ck : "/tmp/srd_needle_ckpt";
    int start_step = 0;
    {
        std::ifstream st(ckpt_dir + "/state.txt");
        if (st >> start_step && start_step > 0) {
            std::printf("resuming from step %d\n", start_step);
            for (auto& lane : lanes)
                lane.model.load_state_dict(load_safetensors(
                    ckpt_dir + "/" + lane.name + ".safetensors"));
        } else {
            start_step = 0;
        }
    }

    std::ofstream train_csv(prefix + "_train.csv",
                            start_step ? std::ios::app : std::ios::out);
    std::ofstream probe_csv(prefix + "_probe.csv",
                            start_step ? std::ios::app : std::ios::out);
    if (!start_step) {
        train_csv << "step,exact,kimi,srd,srd_f\n";
        probe_csv << "step,lane,answer_ce,answer_acc,tail_gate,fill_gate\n";
    }

    std::mt19937 batch_rng(123);
    for (int s = 0; s < start_step * B; ++s) make_seq(T, batch_rng);  // fast-forward

    auto save_ckpt = [&](int step_done) {
        std::system(("mkdir -p " + ckpt_dir).c_str());
        for (auto& lane : lanes)
            save_safetensors(ckpt_dir + "/" + lane.name + ".safetensors",
                             lane.model.state_dict());
        std::ofstream st(ckpt_dir + "/state.txt");
        st << step_done << "\n";
    };

    auto probe_eval = [&](int step) {
        NoGrad ng;
        for (auto& lane : lanes) {
            lane.model.eval();
            double ce = 0, acc = 0, tail_g = 0, fill_g = 0;
            for (const auto& seq : probes) {
                std::vector<int> x(seq.begin(), seq.end() - 1);
                Var logits = lane.model.forward(x);
                // Softmax CE at the final position only.
                const size_t last = T - 1;
                float mx = -1e30f;
                for (int v = 0; v < VOCAB; ++v)
                    mx = std::max(mx, logits->data(last, v));
                double z = 0;
                for (int v = 0; v < VOCAB; ++v)
                    z += std::exp(logits->data(last, v) - mx);
                const int ans = seq[T];
                ce += -(logits->data(last, ans) - mx - std::log(z));
                int arg = 0;
                for (int v = 1; v < VOCAB; ++v)
                    if (logits->data(last, v) > logits->data(last, arg)) arg = v;
                acc += (arg == ans) ? 1.0 : 0.0;
                if (lane.is_srd) {
                    // Gate profile from block 0 of the LAST forward.
                    const Var g = lane.model.srd[0]->gate();
                    tail_g += 0.5 * (g->data(T - 2, 0) + g->data(T - 1, 0));
                    double fg = 0;
                    size_t n = 0;
                    for (size_t t = 20; t < T - 10; ++t, ++n)
                        fg += g->data(t, 0);
                    fill_g += fg / static_cast<double>(n);
                }
            }
            const double N = probes.size();
            probe_csv << step << ',' << lane.name << ',' << ce / N << ','
                      << acc / N << ',' << tail_g / N << ',' << fill_g / N
                      << '\n';
            std::printf("  probe %-5s: answer_ce %.4f acc %.3f", lane.name,
                        ce / N, acc / N);
            if (lane.is_srd)
                std::printf("  tail_gate %.3f fill_gate %.3f", tail_g / N,
                            fill_g / N);
            std::printf("\n");
            lane.model.train();
        }
        probe_csv.flush();
    };

    std::printf("batch=%d npairs=%d nkeys=%d (uniform-CE floor %.4f)\n", B,
                g_npairs, g_nkeys, std::log(static_cast<double>(g_nkeys)));
    std::printf("%5s %9s %9s %9s %9s\n", "step", "exact", "kimi", "srd",
                "srd_f");
    for (int step = start_step + 1; step <= steps; ++step) {
        std::vector<std::vector<int>> batch;
        for (int b = 0; b < B; ++b) batch.push_back(make_seq(T, batch_rng));

        float losses[4];
        for (size_t li = 0; li < lanes.size(); ++li) {
            Lane& lane = lanes[li];
            lane.opt.zero_grad();
            double task_sum = 0;
            for (const auto& seq : batch) {
                std::vector<int> x(seq.begin(), seq.end() - 1);
                std::vector<int> y(seq.begin() + 1, seq.end());
                Var logits = lane.model.forward(x);
                Var task = ops::cross_entropy(logits, y);
                Var loss =
                    lane.is_srd
                        ? ops::add(task, ops::scale(lane.model.mean_gate(),
                                                    lambda_gate))
                        : task;
                backward(ops::scale(loss, 1.0f / static_cast<float>(B)));
                task_sum += task->data(0, 0);
            }
            ops::clip_grad_norm(lane.model.parameters(), 1.0f);
            lane.opt.step();
            losses[li] = static_cast<float>(task_sum / B);
        }
        train_csv << step << ',' << losses[0] << ',' << losses[1] << ','
                  << losses[2] << ',' << losses[3] << '\n';
        if (step % 10 == 0)
            std::printf("%5d %9.4f %9.4f %9.4f %9.4f\n", step, losses[0],
                        losses[1], losses[2], losses[3]);
        if (step % PROBE_EVERY == 0) {
            std::printf("-- probe @ %d --\n", step);
            probe_eval(step);
        }
        std::fflush(stdout);
    }
    save_ckpt(steps);
    std::printf("done through %d; wrote %s_{train,probe}.csv\n", steps,
                prefix.c_str());
    return 0;
}
