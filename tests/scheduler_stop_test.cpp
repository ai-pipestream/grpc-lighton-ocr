#include "engine/inference_control.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

extern "C" void lighton_ggml_set_observer(void (*)(ggml_backend_t, ggml_tensor*, void*), void*);
using namespace lighton;
using namespace lighton::detail;
static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Fake {
  ggml_backend_t base = ggml_backend_cpu_init();
  ggml_backend_device device = *ggml_backend_get_device(base);
  ggml_backend backend = *base;
  int computes = 0, syncs = 0, copies = 0;
  struct Copy { const ggml_tensor* src; ggml_tensor* dst; };
  std::vector<Copy> queued;
  Fake* peer = nullptr;
  ggml_tensor *copy_src = nullptr, *copy_dst = nullptr;
  bool queue_on_compute = false;
  Control* poll_control = nullptr;
  bool expire_on_compute = false;
  Fake() {
    backend.device = &device; backend.context = this;
    backend.iface.graph_compute = [](ggml_backend_t b, ggml_cgraph*) {
      auto& f = *static_cast<Fake*>(b->context); ++f.computes;
      if (f.queue_on_compute) {
        // Model an outstanding cross-backend transfer. Only synchronize writes it.
        ggml_backend_tensor_copy_async(b, &f.peer->backend, f.copy_src, f.copy_dst);
        f.queue_on_compute = false;
      }
      if (f.poll_control) {
        f.poll_control->next_poll = Clock::time_point::min();
        if (f.expire_on_compute) f.poll_control->deadline = Clock::time_point::min();
      }
      return GGML_STATUS_SUCCESS;
    };
    backend.iface.synchronize = [](ggml_backend_t b) {
      auto& f = *static_cast<Fake*>(b->context); ++f.syncs;
      for (auto copy : f.queued) { ggml_backend_tensor_copy(copy.src, copy.dst); ++f.copies; }
      f.queued.clear();
    };
    backend.iface.cpy_tensor_async = [](ggml_backend_t, ggml_backend_t dst, const ggml_tensor* a, ggml_tensor* b) {
      static_cast<Fake*>(dst->context)->queued.push_back({a,b}); return true;
    };
  }
  ~Fake() { ggml_backend_free(base); }
};
struct Params { ggml_backend_sched_eval_callback cb_eval = nullptr; void* cb_eval_user_data = nullptr; };
static void scheduler_case(bool observe, bool vision, bool deadline, bool mid) {
  Fake a,b; a.peer=&b; b.peer=&a;
  a.device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor* t) {
    return std::strcmp(ggml_get_name(t), "stage1") != 0;
  };
  ggml_backend_t backends[]{&a.backend,&b.backend};
  auto sched=ggml_backend_sched_new(backends,nullptr,2,128,false,false);
  EvaluationControl evaluation;
  Params params;
  if (vision) evaluation.configure_vision(params); else evaluation.configure_decoder(params);
  ggml_backend_sched_set_eval_callback(sched,params.cb_eval,params.cb_eval_user_data);
  int observed=0;
  lighton_ggml_set_observer(observe ? +[](ggml_backend_t, ggml_tensor*, void* p) { ++*static_cast<int*>(p); } : nullptr,&observed);
  // Reuse the same scheduler and callback state with a different graph size.
  for (int request=0;request<2;++request) {
    ggml_backend_sched_reset(sched);
    auto ctx=ggml_init({65536,nullptr,true});
    auto input=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,request ? 19 : 8);
    auto copied=ggml_dup_tensor(ctx,input);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,a.base);
    std::vector<float> values(static_cast<size_t>(input->ne[0]),7.f);
    ggml_backend_tensor_set(input,values.data(),0,ggml_nbytes(input));
    ggml_backend_buffer_clear(buffer,0);
    ggml_backend_tensor_set(input,values.data(),0,ggml_nbytes(input));
    auto n0=ggml_sqr(ctx,input);ggml_set_name(n0,"stage0");
    auto n1=ggml_sqr(ctx,n0);ggml_set_name(n1,"stage1");
    auto n2=ggml_sqr(ctx,n1);ggml_set_name(n2,"stage0last");
    auto graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,n2);
    ggml_backend_sched_set_tensor_backend(sched,n0,&a.backend);
    ggml_backend_sched_set_tensor_backend(sched,n1,&b.backend);
    ggml_backend_sched_set_tensor_backend(sched,n2,&a.backend);
    Control control{Clock::now()+std::chrono::hours(1),[&]{return request==0 && !deadline && (!mid || a.computes>0);}};
    if (request==0 && deadline && !mid) control.deadline=Clock::time_point::min();
    evaluation.active=&control;
    require(ggml_backend_sched_alloc_graph(sched,graph),"graph allocation failed");
    require(ggml_backend_sched_get_n_splits(sched)>=3,"test requires multiple backend splits");
    auto& copying = mid ? b : a;
    copying.copy_src=input;copying.copy_dst=copied;copying.queue_on_compute=request==0;
    a.poll_control=&control;a.expire_on_compute=request==0 && deadline && mid;
    a.computes=b.computes=0;
    auto status=ggml_backend_sched_graph_compute_async(sched,graph);
    if (request==0) {
      require(status==GGML_STATUS_ABORTED,"callback stop must abort submission");
      require(a.computes==1 && b.computes==(mid ? 1 : 0),"later splits must not execute");
      require(a.queued.empty() && b.queued.empty() && (mid ? a.copies : b.copies)==1,"all queues drained before caller cleanup");
      float value=0;ggml_backend_tensor_get(copied,&value,0,sizeof(value));
      require(value==7.f,"cross-backend copy must finish before rollback");
      try { control.check(); require(false,"stopped control accepted"); }
      catch(const EngineError& e) { require(e.kind==(deadline?EngineError::deadline:EngineError::cancelled),"wrong stop kind"); }
    } else {
      require(status==GGML_STATUS_SUCCESS && a.computes==2 && b.computes==1,"different-size next request must run all splits");
      ggml_backend_sched_synchronize(sched);
      control.check();
    }
    require(evaluation.vision_active==vision,"production stage callback not used");
    evaluation.active=nullptr; a.poll_control=nullptr;
    ggml_backend_sched_reset(sched);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
  }
  require(observe ? observed>0 : observed==0,"observation off must emit no placement evidence");
  lighton_ggml_set_observer(nullptr,nullptr);
  ggml_backend_sched_free(sched);
}
static void boundary_cases() {
  for(bool observe:{false,true}) {
    int observed=0;
    lighton_ggml_set_observer(observe ? +[](ggml_backend_t, ggml_tensor*, void* p) { ++*static_cast<int*>(p); } : nullptr,&observed);
    for(bool cancelled:{false,true}) {
      bool stop=false;Control control{Clock::now()+std::chrono::hours(1),[&]{return stop;}};
      EvaluationControl evaluation; evaluation.active=&control;
      Params params; evaluation.configure_vision(params);
      bool placement_checked=false;
      try {
        checked_native_call(control,[&] {
          if(cancelled)stop=true;else control.deadline=Clock::time_point::min();
          require(params.cb_eval(nullptr,true,params.cb_eval_user_data),"stop callback missing");
          require(!params.cb_eval(nullptr,false,params.cb_eval_user_data),"completion must abort");
          return 0; // Fake native success despite the callback stop, no matmul.
        });
        placement_checked=true; // Fake success with no matmul must stop before placement validation.
      } catch(const EngineError& e) { require(e.kind==(cancelled?EngineError::cancelled:EngineError::deadline),"success status masked stop"); }
      require(!placement_checked,"missing-placement error won over stop");
    }
    require(observed==0,"no-matmul call must not invent placement");
    lighton_ggml_set_observer(nullptr,nullptr);
    for(bool complete:{false,true}) {
      Control control{Clock::now()+std::chrono::hours(1),[]{return false;}};
      try {
        // max_tokens=1: successful final decode that expires before returning.
        checked_native_call(control,[&]{control.deadline=Clock::time_point::min();return 0;});
        require(false,"final decode accepted expired success");
      }catch(const EngineError& e){require(e.kind==EngineError::deadline,"final decode deadline lost");}
      try { (void)terminal_finish(control,complete);require(false,"terminal result masked deadline"); }
      catch(const EngineError& e){require(e.kind==EngineError::deadline,"terminal deadline lost");}
    }
  }
}
int main() {
  try {
    for(bool observe:{false,true})for(bool vision:{false,true})for(bool deadline:{false,true})for(bool mid:{false,true})scheduler_case(observe,vision,deadline,mid);
    boundary_cases();
    std::cout<<"PASS scheduler abort/drain/reuse, observation on/off, vision/decoder, cancellation/deadline, success and final-token boundaries\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
