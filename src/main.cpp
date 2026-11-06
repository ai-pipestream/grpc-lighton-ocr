#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "config.h"
#include "service/lighton_service.h"

namespace {

int g_shutdown_pipe[2] = {-1, -1};

void handle_shutdown(int /*signum*/) {
  const int saved_errno = errno;
  const char one = 1;
  // A full pipe already records shutdown. Never block inside a signal handler.
  [[maybe_unused]] ssize_t n = ::write(g_shutdown_pipe[1], &one, 1);
  errno = saved_errno;
}

void install_shutdown_pipe() {
  if (::pipe2(g_shutdown_pipe, O_CLOEXEC) != 0) {
    throw std::runtime_error("shutdown pipe creation failed");
  }
  // Keep the reader blocking, but make repeated signals safe even if the
  // process has not reached its shutdown read yet.
  if (::fcntl(g_shutdown_pipe[1], F_SETFL, O_NONBLOCK) == -1) {
    throw std::runtime_error("shutdown pipe nonblocking setup failed");
  }
  struct sigaction action = {};
  action.sa_handler = handle_shutdown;
  if (::sigaction(SIGINT, &action, nullptr) != 0 || ::sigaction(SIGTERM, &action, nullptr) != 0) {
    throw std::runtime_error("shutdown signal handler install failed");
  }
  struct sigaction ignore = {};
  ignore.sa_handler = SIG_IGN;
  if (::sigaction(SIGPIPE, &ignore, nullptr) != 0) {
    throw std::runtime_error("SIGPIPE ignore failed");
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    install_shutdown_pipe();
    const lighton::Config config = lighton::load_config_from_env();
    lighton::LightOnOcrServiceImpl service(config);

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(config.listen_address, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (server == nullptr) {
      std::cerr << "Unable to listen on " << config.listen_address << '\n';
      return 1;
    }
    auto* health = server->GetHealthCheckService();
    health->SetServingStatus(lighton::locrv1::LightOnOcrService::service_full_name(),
                             service.engine_loaded());
    // The empty health service name remains transport liveness. Callers that
    // route OCR requests must check the fully qualified service name.
    std::cout << "grpc-lighton-ocr listening on " << config.listen_address << '\n';

    char byte = 0;
    while (::read(g_shutdown_pipe[0], &byte, 1) < 0 && errno == EINTR) {
    }
    health->SetServingStatus(false);
    server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
    server->Wait();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
