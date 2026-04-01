#pragma once
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include "packet_parser.hpp"

class EventSender {
public:
    explicit EventSender(const std::string& socket_path)
        : socket_path_(socket_path), fd_(-1)
    {}

    ~EventSender() {
        if (fd_ >= 0) close(fd_);
    }

    // Блокирующее подключение — ждём пока Python поднимет сокет
    void connect() {
        fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) throw std::runtime_error("socket() failed");

        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

        for (int attempt = 0; attempt < 30; ++attempt) {
            if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
                return;
            sleep(1);
        }

        throw std::runtime_error("Could not connect to " + socket_path_);
    }

    // Отправка одного события — write атомарен для маленьких размеров
    bool send(const NetworkEventMsg& msg) {
        ssize_t n = ::write(fd_, &msg, sizeof(msg));
        return n == static_cast<ssize_t>(sizeof(msg));
    }

private:
    std::string socket_path_;
    int fd_;
};
