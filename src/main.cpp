#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <csignal>
#include <iostream>
#include <memory>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "config.h"
#include "service/lighton_service.h"

namespace {

int g_shutdown_pipe[2] = {-1, -1};

void handle_shutdown(int /*signum*/) {
  char one = 1;
  [[maybe_unused]] ssize_t n = ::write(g_shutdown_pipe[1], &one, 1);
}

void install_shutdown_pipe() {
  if (::pipe2(g_shutdown_pipe, O_CLOEXEC) != 0) {
    throw std::runtime_error("shutdown pipe creation failed");
  }
  struct sigaction action = {};
  action.sa_handler = handle_shutdown;
  ::sigaction(SIGINT, &action, nullptr);
  ::sigaction(SIGTERM, &action, nullptr);
  ::signal(SIGPIPE, SIG_IGN);
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
    std::cout << "grpc-lighton-ocr listening on " << config.listen_address << '\n';

    char byte = 0;
    while (::read(g_shutdown_pipe[0], &byte, 1) < 0 && errno == EINTR) {
    }
    server->Shutdown();
    server->Wait();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
