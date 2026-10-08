#include "lamina/model/inference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        for (const char* mode : {"f32", "fast"}) {
            lamina::model::Inference model(argv[1], 4096, 40, true, "device", 0, "f16", mode);
            model.prefill_hidden({42,43,44,45,46,47,48,49});
            model.cache_prefix();
            const auto expected = model.prefill_hidden({50,51,52,53});
            model.greedy(expected);
            for (int i=0; i<6; ++i) model.step_hidden(60+i);
            model.restore_prefix();
            if (model.position()!=8 || model.rope_position()!=8) throw std::runtime_error("position restore failed");
            auto actual=model.prefill_hidden({50,51,52,53});
            double error=0;
            for(size_t i=0;i<actual.size();++i) error=std::max(error,std::abs(double(actual[i])-expected[i]));
            std::printf("%s restored suffix max_abs_diff=%.9g\n",mode,error);
            if(error>=1e-5) throw std::runtime_error("prefix state differs");
            model.greedy(actual);
            model.restore_prefix();
            model.prefill_long(std::vector<int>(65,75),32);
            model.restore_prefix();
            actual=model.prefill_hidden({50,51,52,53});
            for(size_t i=0;i<actual.size();++i) if(std::abs(actual[i]-expected[i])>=1e-5) throw std::runtime_error("long suffix corrupted prefix");
            model.reset();
            bool rejected=false;
            try { model.restore_prefix(); } catch(const std::exception&) { rejected=true; }
            if(!rejected) throw std::runtime_error("RESET retained stale prefix");
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
