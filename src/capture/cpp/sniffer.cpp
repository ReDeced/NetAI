#include <pcap.h>
#include <csignal>
#include <cstring>
#include <iostream>
#include <atomic>
#include "packet_parser.hpp"
#include "event_sender.hpp"

static std::atomic<bool> g_running{true};
static EventSender* g_sender = nullptr;

void signal_handler(int) {
    g_running = false;
}

void packet_handler(
    u_char* user,
    const pcap_pkthdr* header,
    const u_char* bytes
) {
    auto* sender = reinterpret_cast<EventSender*>(user);

    NetworkEventMsg msg{};
    double ts = header->ts.tv_sec + header->ts.tv_usec * 1e-6;

    if (!parse_packet(bytes, header->caplen, ts, msg)) return;

    if (!sender->send(msg)) {
        std::cerr << "[sniffer] send failed, receiver gone?\n";
        g_running = false;
    }
}

int main(int argc, char* argv[]) {
    const char* iface       = argc > 1 ? argv[1] : "br0";
    const char* socket_path = argc > 2 ? argv[2] : "/tmp/netai.sock";
    const char* bpf_filter  = argc > 3 ? argv[3] : "ip";

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // Подключаемся к Python-приёмнику
    EventSender sender(socket_path);
    std::cout << "[sniffer] Connecting to " << socket_path << "...\n";
    sender.connect();
    std::cout << "[sniffer] Connected.\n";
    g_sender = &sender;

    // Открываем pcap
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* handle = pcap_open_live(iface, 65535, 1, 1, errbuf);
    if (!handle) {
        std::cerr << "[sniffer] pcap_open_live: " << errbuf << "\n";
        return 1;
    }

    // Устанавливаем BPF-фильтр
    bpf_program fp{};
    if (pcap_compile(handle, &fp, bpf_filter, 0, PCAP_NETMASK_UNKNOWN) < 0 ||
        pcap_setfilter(handle, &fp) < 0) {
        std::cerr << "[sniffer] BPF filter error: " << pcap_geterr(handle) << "\n";
        pcap_close(handle);
        return 1;
    }

    std::cout << "[sniffer] Capturing on " << iface << " ...\n";

    // Главный цикл — неблокирующий опрос
    while (g_running) {
        int ret = pcap_dispatch(handle, 64, packet_handler,
                                reinterpret_cast<u_char*>(&sender));
        if (ret < 0) break;
    }

    pcap_freecode(&fp);
    pcap_close(handle);
    std::cout << "[sniffer] Stopped.\n";
    return 0;
}
